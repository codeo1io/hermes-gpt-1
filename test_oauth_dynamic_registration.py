"""RFC 7591 dynamic client registration + public-client self-heal.

Reproduces the live 2026-09-06 18:42 connector outage: OpenAI's platform
attempted headless re-auth with an ephemeral client_id (401 "Unknown OAuth
client") and called POST /oauth/register (401 — endpoint absent), while the
scheme-flow credential carried no refresh token (scope lacked
offline_access) so the connector could never self-heal.
"""

from __future__ import annotations

import base64
import hashlib

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from oauth_auth import (
    OAuthConfig,
    OAuthState,
    authorize,
    register_client,
    token,
)

CLIENT_ID = "chatgpt-client"
CLIENT_SECRET = "test-client-secret-0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ"
REDIRECT_URI = "https://chatgpt.com/connector/oauth/callback"
ISSUER = "https://mcp.example.com"

VERIFIER = "b" * 64
CHALLENGE_S256 = (
    base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest())
    .rstrip(b"=")
    .decode()
)

AIP_REDIRECT = "https://chatgpt.com/aip/oauth/callback"


def _config() -> OAuthConfig:
    return OAuthConfig(
        issuer=ISSUER,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uris=(REDIRECT_URI,),
        scope="hermes",
    )


def _build(state: OAuthState) -> TestClient:
    async def authorize_ep(request: Request) -> JSONResponse:
        return authorize(request, state)

    async def token_ep(request: Request) -> JSONResponse:
        return await token(request, state)

    async def register_ep(request: Request) -> JSONResponse:
        return await register_client(request, state)

    app = Starlette(
        routes=[
            Route("/oauth/authorize", authorize_ep, methods=["GET"]),
            Route("/oauth/token", token_ep, methods=["POST"]),
            Route("/oauth/register", register_ep, methods=["POST"]),
        ]
    )
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def state() -> OAuthState:
    return OAuthState(_config())


@pytest.fixture
def client(state: OAuthState) -> TestClient:
    return _build(state)


def _register(client: TestClient, **overrides):
    payload = {
        "client_name": "OpenAI apps flow",
        "redirect_uris": [AIP_REDIRECT],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
    }
    payload.update(overrides)
    return client.post("/oauth/register", json=payload)


def _authorize(client: TestClient, client_id: str, redirect: str, scope: str = "hermes"):
    return client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect,
            "scope": scope,
            "state": "s1",
            "code_challenge": CHALLENGE_S256,
            "code_challenge_method": "S256",
        },
    )


def _exchange(client: TestClient, code: str, redirect: str, client_id: str = ""):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect,
        "code_verifier": VERIFIER,
    }
    if client_id:
        data["client_id"] = client_id
    return client.post("/oauth/token", data=data)


# -- registration endpoint -------------------------------------------------


def test_register_mints_dynamic_public_client(client: TestClient):
    response = _register(client)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["client_id"] != CLIENT_ID
    assert body["client_id_issued_at"] > 0
    assert "client_secret" not in body
    assert body["redirect_uris"] == [AIP_REDIRECT]
    assert body["token_endpoint_auth_method"] == "none"


def test_register_refuses_non_chatgpt_redirects(client: TestClient):
    response = _register(client, redirect_uris=["https://evil.example.com/cb"])
    assert response.status_code == 400
    assert "redirect" in response.json()["error_description"].lower()


def test_register_refuses_confidential_auth_method(client: TestClient):
    response = _register(client, token_endpoint_auth_method="client_secret_post")
    assert response.status_code == 400


def test_register_rejects_malformed_payload(client: TestClient):
    response = client.post("/oauth/register", data="not-json")
    assert response.status_code == 400


def test_register_capacity_is_bounded(client: TestClient, state: OAuthState):
    codes = [_register(client).status_code for _ in range(state.max_dynamic_clients + 2)]
    assert 429 in codes


# -- dynamic client authorize + exchange ------------------------------------


def test_dynamic_client_authorizes_with_pkce(client: TestClient):
    dyn_id = _register(client).json()["client_id"]
    response = _authorize(client, dyn_id, AIP_REDIRECT)
    assert response.status_code == 302, response.text
    assert "code=" in response.headers["location"]
    assert "error" not in response.headers["location"]


def test_dynamic_client_redirect_must_match_registered(client: TestClient):
    dyn_id = _register(client).json()["client_id"]
    response = _authorize(client, dyn_id, REDIRECT_URI)  # not its registered one
    # error redirect per RFC (the client's redirect_uri is unknown to us, so
    # the static config allowlist catches it); never a usable code
    assert response.status_code in (302, 400)
    body = response.headers.get("location", "") + response.text
    assert "error" in body


def test_unknown_client_still_refused(client: TestClient):
    response = _authorize(client, "totally-unknown", AIP_REDIRECT)
    assert response.status_code == 401


def test_dynamic_client_full_flow_self_heals(client: TestClient):
    dyn_id = _register(client).json()["client_id"]
    auth = _authorize(client, dyn_id, AIP_REDIRECT, scope="hermes offline_access")
    assert auth.status_code == 302, auth.text
    code = auth.headers["location"].split("code=")[1].split("&")[0]
    tok = _exchange(client, code, AIP_REDIRECT, client_id=dyn_id)
    assert tok.status_code == 200, tok.text
    body = tok.json()
    assert body.get("refresh_token"), "offline_access must yield a refresh token"

    # self-heal: the refresh grant works secretless for a public client
    refresh = client.post(
        "/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": body["refresh_token"],
            "client_id": dyn_id,
        },
    )
    assert refresh.status_code == 200, refresh.text
    rotated = refresh.json()
    assert rotated["access_token"]
    assert rotated.get("refresh_token")


def test_dynamic_client_without_offline_access_still_gets_refresh_token(client: TestClient):
    """Every code exchange now yields a refresh token so a dying credential
    can self-heal (the 18:42 outage root cause)."""
    dyn_id = _register(client).json()["client_id"]
    auth = _authorize(client, dyn_id, AIP_REDIRECT, scope="hermes")
    code = auth.headers["location"].split("code=")[1].split("&")[0]
    tok = _exchange(client, code, AIP_REDIRECT, client_id=dyn_id)
    assert tok.status_code == 200, tok.text
    assert tok.json().get("refresh_token"), "refresh token must always be issued"


def test_dynamic_client_pkce_required(client: TestClient):
    dyn_id = _register(client).json()["client_id"]
    response = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": dyn_id,
            "redirect_uri": AIP_REDIRECT,
            "scope": "hermes",
        },
    )
    # error redirect to the registered redirect_uri; never a usable code
    assert response.status_code == 302
    assert "error=" in response.headers["location"]


def test_dynamic_client_registry_survives_state_serialization(client: TestClient, state: OAuthState):
    dyn_id = _register(client).json()["client_id"]
    bundle = state.export_dynamic_clients()
    state2 = OAuthState(_config())
    state2.import_dynamic_clients(bundle)
    response = _authorize(_build(state2), dyn_id, AIP_REDIRECT)
    assert "code=" in response.headers["location"]


# -- legacy static client unchanged ------------------------------------------


def test_legacy_static_client_flow_unchanged(client: TestClient):
    """chatgpt-client confidential + optional-PKCE compat is untouched: in
    optional mode a challenge-less authorize still issues a code (the
    confidential redemption is enforced at the token endpoint)."""
    state_opt = OAuthState(
        OAuthConfig(
            issuer=ISSUER,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uris=(REDIRECT_URI,),
            scope="hermes",
            pkce_mode="optional",
        )
    )
    legacy = _build(state_opt)
    auth = legacy.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": "hermes",
            "state": "legacy",
        },
    )
    assert auth.status_code == 302, auth.text
    assert "code=" in auth.headers["location"]
