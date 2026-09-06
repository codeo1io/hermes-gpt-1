from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

AUTH_CODE_TTL_SECONDS = 300
ACCESS_TOKEN_TTL_SECONDS = 30 * 24 * 3600
REFRESH_TOKEN_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_AUTH_CODES = 1024
MAX_ACCESS_TOKENS = 4096
MAX_REFRESH_TOKENS = 4096
MAX_TOKEN_REQUEST_BYTES = 16384
AUTH_TOKEN_ENV = "HERMES_GPT_BEARER_TOKEN"
OAUTH_ENABLE_ENV = "HERMES_GPT_OAUTH_ENABLE"
OAUTH_ISSUER_ENV = "HERMES_GPT_OAUTH_ISSUER"
OAUTH_CLIENT_ID_ENV = "HERMES_GPT_OAUTH_CLIENT_ID"
OAUTH_CLIENT_SECRET_ENV = "HERMES_GPT_OAUTH_CLIENT_SECRET"
OAUTH_REDIRECT_URI_ENV = "HERMES_GPT_OAUTH_REDIRECT_URI"
OAUTH_SCOPE_ENV = "HERMES_GPT_OAUTH_SCOPE"
OAUTH_PKCE_MODE_ENV = "HERMES_GPT_OAUTH_PKCE_MODE"
_PKCE_MODES = ("required", "optional")
_PKCE_VALUE = re.compile(r"^[A-Za-z0-9._~-]{43,128}$")
_CLIENT_SECRET = re.compile(r"^[A-Za-z0-9._~-]{43,128}$")
_BASE64URL = re.compile(r"^[A-Za-z0-9_-]+$")
_NONCE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")

# Optional persistence hook (v0.7 S5). server.py installs it so every token
# issuance/refresh persists through token_store without oauth_auth depending
# on a concrete hermes_root. Never raises; token material never logged.
_persist_hook: Any | None = None


def set_persist_hook(hook: Any | None) -> None:
    """Install (or clear) the durable-token persistence hook.

    The hook receives ``(state, kind)`` where kind is one of
    ``authorization_code`` | ``refresh`` and returns a bounded summary.
    """
    global _persist_hook
    _persist_hook = hook


def _run_persist_hook(state: "OAuthState", kind: str) -> None:
    if _persist_hook is None:
        return
    try:
        _persist_hook(state, kind)
    except Exception:
        # Persistence must never break the token exchange path.
        pass


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    if not value or not _BASE64URL.fullmatch(value):
        raise ValueError("invalid base64url value")
    padding = "=" * (-len(value) % 4)
    decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    if _base64url_encode(decoded) != value:
        raise ValueError("non-canonical base64url value")
    return decoded


#: Hosts dynamic clients may redirect to (RFC 7591 allowlist). The redirect
#: allowlist is the load-bearing control: dynamic clients are public (PKCE
#: only, no secret), so the redirect_uri is the only party-binding we get.
_DYNAMIC_CLIENT_REDIRECT_HOSTS = frozenset(
    {"chatgpt.com", "chat.openai.com", "openai.com"}
)

#: Registration endpoint body bound (RFC 7591 payloads are small).
_MAX_REGISTER_BYTES = 8 * 1024

#: Bounded dynamic-client registry (single-tenant issuer).
MAX_DYNAMIC_CLIENTS = 64


class OAuthError(RuntimeError):
    def __init__(self, error: str, description: str, *, status_code: int = 400) -> None:
        super().__init__(description)
        self.error = error
        self.description = description
        self.status_code = status_code


@dataclass(frozen=True)
class OAuthConfig:
    issuer: str
    client_id: str
    client_secret: str
    redirect_uris: tuple[str, ...]
    scope: str = "hermes"
    pkce_mode: str = "required"

    def __post_init__(self) -> None:
        if self.pkce_mode not in _PKCE_MODES:
            raise ValueError(f"pkce_mode must be one of {_PKCE_MODES}: {self.pkce_mode!r}")
        issuer = self.issuer.rstrip("/")
        parsed = urllib.parse.urlparse(issuer)
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("OAuth issuer must use HTTPS except on loopback.")
        if (
            not parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("OAuth issuer must be an origin URL without path, userinfo, query, or fragment.")
        if not self.client_id.strip():
            raise ValueError("OAuth client_id is required.")
        if not _CLIENT_SECRET.fullmatch(self.client_secret):
            raise ValueError("OAuth client_secret must contain 43 to 128 URL-safe characters.")
        if not self.redirect_uris:
            raise ValueError("At least one OAuth redirect URI is required.")
        for redirect_uri in self.redirect_uris:
            redirect = urllib.parse.urlparse(redirect_uri)
            if (
                redirect.scheme != "https"
                or not redirect.netloc
                or not redirect.hostname
                or redirect.fragment
                or redirect.username is not None
                or redirect.password is not None
            ):
                raise ValueError("OAuth redirect URIs must be absolute HTTPS URLs without userinfo or fragments.")
            if redirect_uri.endswith("*"):
                prefix = redirect_uri[:-1]
                if not prefix:
                    raise ValueError("OAuth redirect URI wildcard must follow a non-empty prefix.")
                prefix_parsed = urllib.parse.urlparse(prefix)
                if (
                    prefix_parsed.scheme != "https"
                    or not prefix_parsed.netloc
                    or not prefix_parsed.hostname
                ):
                    raise ValueError("OAuth redirect URI wildcard prefix must be an HTTPS origin/prefix.")
                if "*" in prefix:
                    raise ValueError("Only a single trailing wildcard is allowed in OAuth redirect URIs.")
        if not self.scope.strip() or len(self.scope.split()) != 1:
            raise ValueError("OAuth scope must be one non-empty scope token.")
        object.__setattr__(self, "issuer", issuer)
        object.__setattr__(self, "client_id", self.client_id.strip())
        object.__setattr__(self, "redirect_uris", tuple(dict.fromkeys(self.redirect_uris)))
        object.__setattr__(self, "scope", self.scope.strip())

    @property
    def resource(self) -> str:
        return f"{self.issuer}/mcp"

    @property
    def supported_scopes(self) -> tuple[str, ...]:
        # ChatGPT currently adds `openid` even when its OIDC toggle is disabled.
        # It is accepted as a compatibility scope; this server does not advertise
        # OpenID Provider metadata or issue ID tokens.
        return tuple(dict.fromkeys((self.scope, "openid", "offline_access")))


@dataclass(frozen=True)
class DynamicClient:
    """A dynamically registered public OAuth client (RFC 7591).

    Public client: PKCE only, no secret. The redirect allowlist is the
    party-binding control (chatgpt.com family only).
    """

    client_id: str
    redirect_uris: tuple[str, ...]
    registered_at: float

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "client_id_issued_at": int(self.registered_at),
            "redirect_uris": list(self.redirect_uris),
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
        }


def _registerable_redirect(redirect_uri: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(redirect_uri)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and host in _DYNAMIC_CLIENT_REDIRECT_HOSTS
        and not parsed.fragment
        and parsed.username is None
        and parsed.password is None
    )


class OAuthState:
    def __init__(
        self,
        config: OAuthConfig,
        *,
        max_auth_codes: int = MAX_AUTH_CODES,
        max_access_tokens: int = MAX_ACCESS_TOKENS,
        max_refresh_tokens: int = MAX_REFRESH_TOKENS,
        max_dynamic_clients: int = MAX_DYNAMIC_CLIENTS,
    ) -> None:
        self.config = config
        self.max_auth_codes = max_auth_codes
        self.max_access_tokens = max_access_tokens
        self.max_refresh_tokens = max_refresh_tokens
        self.max_dynamic_clients = max_dynamic_clients
        self._authorization_code_key = secrets.token_bytes(32)
        self.used_auth_codes: dict[str, dict[str, Any]] = {}
        self.access_tokens: dict[str, dict[str, Any]] = {}
        self.refresh_tokens: dict[str, dict[str, Any]] = {}
        self.dynamic_clients: dict[str, DynamicClient] = {}

    def register_dynamic_client(self, redirect_uris: list[str]) -> DynamicClient:
        """Mint a public dynamic client; bounded, chatgpt.com-redirects only."""
        cleaned = list(dict.fromkeys(redirect_uris))
        if not cleaned or len(cleaned) > 8:
            raise OAuthError(
                "invalid_client_metadata",
                "Between 1 and 8 redirect_uris are required.",
            )
        for redirect_uri in cleaned:
            if not isinstance(redirect_uri, str) or not _registerable_redirect(redirect_uri):
                raise OAuthError(
                    "invalid_redirect_uri",
                    "redirect_uris must be absolute HTTPS URLs on chatgpt.com/openai.com hosts.",
                )
        if len(self.dynamic_clients) >= self.max_dynamic_clients:
            raise OAuthError(
                "temporarily_unavailable",
                "Dynamic client capacity is temporarily unavailable.",
                status_code=429,
            )
        client_id = f"dyn-{secrets.token_urlsafe(12)}"
        client = DynamicClient(
            client_id=client_id,
            redirect_uris=tuple(cleaned),
            registered_at=time.time(),
        )
        self.dynamic_clients[client_id] = client
        return client

    def client_redirect_allowed(self, client_id: str, redirect_uri: str) -> bool:
        """Redirect check for dynamic clients (exact-match against their
        registered URIs; the static config path keeps its own rules)."""
        client = self.dynamic_clients.get(client_id)
        if client is None:
            return False
        return redirect_uri in client.redirect_uris

    def is_dynamic_client(self, client_id: str) -> bool:
        return client_id in self.dynamic_clients

    def export_dynamic_clients(self) -> list[dict[str, Any]]:
        return [
            {
                "client_id": c.client_id,
                "redirect_uris": list(c.redirect_uris),
                "registered_at": c.registered_at,
            }
            for c in self.dynamic_clients.values()
        ]

    def import_dynamic_clients(self, payload: Any) -> int:
        if not isinstance(payload, list):
            return 0
        imported = 0
        for item in payload:
            if not isinstance(item, dict):
                continue
            client_id = item.get("client_id")
            redirects = item.get("redirect_uris")
            registered_at = item.get("registered_at", 0)
            if not isinstance(client_id, str) or not isinstance(redirects, list):
                continue
            if len(self.dynamic_clients) >= self.max_dynamic_clients:
                break
            self.dynamic_clients[client_id] = DynamicClient(
                client_id=client_id,
                redirect_uris=tuple(str(u) for u in redirects),
                registered_at=float(registered_at or 0),
            )
            imported += 1
        return imported


    def cleanup(self) -> None:
        now = time.time()
        for store in (self.used_auth_codes, self.access_tokens, self.refresh_tokens):
            for credential, item in list(store.items()):
                if item.get("expires_at", 0) <= now:
                    store.pop(credential, None)

    def normalize_scope(self, scope: str) -> str:
        requested = list(dict.fromkeys(scope.split()))
        if (
            not requested
            or self.config.scope not in requested
            or not set(requested).issubset(self.config.supported_scopes)
        ):
            raise OAuthError("invalid_scope", "Requested scope is not supported.")
        return " ".join(requested)

    def _require_capacity(self, store: dict[str, Any], maximum: int, credential_type: str) -> None:
        self.cleanup()
        if len(store) >= maximum:
            raise OAuthError(
                "temporarily_unavailable",
                f"{credential_type} capacity is temporarily unavailable.",
                status_code=503,
            )

    def issue_authorization_code(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        scope: str,
        resource: str,
        code_challenge: str,
    ) -> str:
        payload = {
            "v": 1,
            "nonce": secrets.token_urlsafe(24),
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "resource": resource,
            "code_challenge": code_challenge,
            "expires_at": int(time.time()) + AUTH_CODE_TTL_SECONDS,
        }
        encoded = _base64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        signature = hmac.new(self._authorization_code_key, encoded.encode("ascii"), hashlib.sha256).digest()
        return f"{encoded}.{_base64url_encode(signature)}"

    def _decode_authorization_code(self, code: str) -> dict[str, Any]:
        if len(code) > 4096:
            raise OAuthError("invalid_grant", "Invalid, expired, or already used authorization code.")
        encoded, separator, encoded_signature = code.partition(".")
        if not separator:
            raise OAuthError("invalid_grant", "Invalid, expired, or already used authorization code.")
        try:
            supplied_signature = _base64url_decode(encoded_signature)
            expected_signature = hmac.new(
                self._authorization_code_key,
                encoded.encode("ascii"),
                hashlib.sha256,
            ).digest()
            if not hmac.compare_digest(supplied_signature, expected_signature):
                raise ValueError("signature mismatch")
            payload = json.loads(_base64url_decode(encoded))
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise OAuthError("invalid_grant", "Invalid, expired, or already used authorization code.") from exc
        required_types = {
            "v": int,
            "nonce": str,
            "client_id": str,
            "redirect_uri": str,
            "scope": str,
            "resource": str,
            "code_challenge": str,
            "expires_at": int,
        }
        if not isinstance(payload, dict) or any(not isinstance(payload.get(key), kind) for key, kind in required_types.items()):
            raise OAuthError("invalid_grant", "Invalid, expired, or already used authorization code.")
        if payload["v"] != 1 or not _NONCE.fullmatch(payload["nonce"]):
            raise OAuthError("invalid_grant", "Invalid, expired, or already used authorization code.")
        return payload

    def _new_access_token(self, *, client_id: str, scope: str, resource: str) -> tuple[str, dict[str, Any]]:
        token_value = secrets.token_urlsafe(48)
        item = {
            "client_id": client_id,
            "scope": scope,
            "resource": resource,
            "expires_at": time.time() + ACCESS_TOKEN_TTL_SECONDS,
        }
        return token_value, item

    def _new_refresh_token(self, *, client_id: str, scope: str) -> tuple[str, dict[str, Any]]:
        token_value = secrets.token_urlsafe(48)
        item = {
            "client_id": client_id,
            "scope": scope,
            "expires_at": time.time() + REFRESH_TOKEN_TTL_SECONDS,
        }
        return token_value, item

    def exchange_authorization_code(
        self,
        *,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_verifier: str,
        confidential: bool = False,
    ) -> dict[str, Any]:
        self.cleanup()
        item = self._decode_authorization_code(code)
        nonce = item["nonce"]
        if nonce in self.used_auth_codes or item.get("expires_at", 0) <= time.time():
            raise OAuthError("invalid_grant", "Invalid, expired, or already used authorization code.")
        if item.get("client_id") != client_id or item.get("redirect_uri") != redirect_uri:
            raise OAuthError("invalid_grant", "Authorization code validation failed.")
        challenge = item.get("code_challenge", "")
        # PKCE verification is mode-dependent. "required" (default): a code
        # without a stored S256 challenge (e.g. issued before PKCE
        # enforcement, or by a bypassed authorize path) can never be exchanged
        # — fail closed instead of skipping verification. "optional": a
        # challenge-less code (pre-2026-08-31 compat) is only redeemable by a
        # confidential client — the token endpoint authenticates it with the
        # client_secret (see _authenticate_client / the ``confidential``
        # argument); a public/secretless client must always redeem via PKCE.
        if not challenge and (self.config.pkce_mode != "optional" or not confidential):
            raise OAuthError("invalid_grant", "Authorization code validation failed.")
        if challenge:
            if not _valid_pkce_verifier(code_verifier) or not hmac.compare_digest(_s256(code_verifier), challenge):
                raise OAuthError("invalid_grant", "Authorization code validation failed.")

        scope = self.normalize_scope(item["scope"])
        self._require_capacity(self.used_auth_codes, self.max_auth_codes, "Authorization-code replay cache")
        self._require_capacity(self.access_tokens, self.max_access_tokens, "Access-token")
        self._require_capacity(self.refresh_tokens, self.max_refresh_tokens, "Refresh-token")

        access_value, access_item = self._new_access_token(
            client_id=client_id,
            scope=scope,
            resource=item["resource"],
        )
        response: dict[str, Any] = {
            "access_token": access_value,
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_TTL_SECONDS,
            "scope": scope,
        }
        # 2026-09-06 outage fix: EVERY code exchange now issues a refresh
        # token. The connector's scheme flow authorizes scope=hermes (no
        # offline_access), so its credential could never self-heal after a
        # client-side eviction — the connector then 401-loops forever.
        # Refresh tokens are bound to the client and rotated on every use;
        # issuing one unconditionally cannot escalate scope (a narrowed
        # refresh request is still subset-checked at exchange time).
        refresh_value, refresh_item = self._new_refresh_token(client_id=client_id, scope=scope)
        response["refresh_token"] = refresh_value

        self.used_auth_codes[nonce] = {"expires_at": item["expires_at"]}
        self.access_tokens[access_value] = access_item
        self.refresh_tokens[refresh_value] = refresh_item
        _run_persist_hook(self, "authorization_code")
        return response

    def exchange_refresh_token(
        self,
        *,
        refresh_token: str,
        client_id: str,
        requested_scope: str,
    ) -> dict[str, Any]:
        self.cleanup()
        item = self.refresh_tokens.get(refresh_token)
        if not item or item.get("expires_at", 0) <= time.time():
            self.refresh_tokens.pop(refresh_token, None)
            raise OAuthError("invalid_grant", "Invalid, expired, or already used refresh token.")
        if item.get("client_id") != client_id:
            raise OAuthError("invalid_grant", "Refresh token validation failed.")
        original_scope = self.normalize_scope(item["scope"])
        scope = self.normalize_scope(requested_scope) if requested_scope.strip() else original_scope
        if not set(scope.split()).issubset(original_scope.split()):
            raise OAuthError("invalid_scope", "Requested scope exceeds the originally granted scope.")

        self._require_capacity(self.access_tokens, self.max_access_tokens, "Access-token")
        access_value, access_item = self._new_access_token(
            client_id=client_id,
            scope=scope,
            resource=self.config.resource,
        )
        rotated_value, rotated_item = self._new_refresh_token(client_id=client_id, scope=scope)
        self.refresh_tokens.pop(refresh_token, None)
        self.refresh_tokens[rotated_value] = rotated_item
        self.access_tokens[access_value] = access_item
        _run_persist_hook(self, "refresh")
        return {
            "access_token": access_value,
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_TTL_SECONDS,
            "refresh_token": rotated_value,
            "scope": scope,
        }

    def validate_access_token(self, token_value: str) -> bool:
        if not token_value:
            return False
        self.cleanup()
        item = self.access_tokens.get(token_value)
        return bool(
            item
            and item.get("expires_at", 0) > time.time()
            and item.get("resource") == self.config.resource
        )

    # ------------------------------------------------------------------
    # Durable token persistence (v0.7 S5, ADR-001). Tokens are persisted
    # through token_store (AES-256-GCM envelope); no token material is
    # ever written to the audit log or returned on surfaces.
    # ------------------------------------------------------------------

    def persist_tokens(self, hermes_root: Path | None = None) -> dict[str, Any]:
        """Encrypt + persist the current access/refresh token stores."""
        import token_store

        bundle: dict[str, Any] = {}
        for kind, store in (("access_tokens", self.access_tokens), ("refresh_tokens", self.refresh_tokens)):
            bundle[kind] = {
                value: item
                for value, item in store.items()
                if item.get("expires_at", 0) > time.time()
            }
        bundle["dynamic_clients"] = self.export_dynamic_clients()
        if not hermes_root:
            hermes_root = Path.home() / ".hermes"
        return token_store.save_tokens(hermes_root, bundle)

    def restore_tokens(self, hermes_root: Path | None = None) -> dict[str, Any]:
        """Load + decrypt persisted tokens into the in-memory stores.

        Returns a bounded summary; never exposes token material.
        """
        import token_store

        if not hermes_root:
            hermes_root = Path.home() / ".hermes"
        bundle = token_store.load_tokens(hermes_root)
        if not bundle:
            return {"restored": 0, "present": False}
        restored = 0
        for kind, store in (("access_tokens", self.access_tokens), ("refresh_tokens", self.refresh_tokens)):
            for value, item in (bundle.get(kind) or {}).items():
                if isinstance(item, dict) and item.get("expires_at", 0) > time.time():
                    store[value] = item
                    restored += 1
        restored += self.import_dynamic_clients(bundle.get("dynamic_clients"))
        return {"restored": restored, "present": True}


def config_from_env() -> OAuthConfig | None:
    if os.environ.get(OAUTH_ENABLE_ENV) != "1":
        return None
    required = {
        OAUTH_ISSUER_ENV: os.environ.get(OAUTH_ISSUER_ENV, "").strip(),
        OAUTH_CLIENT_ID_ENV: os.environ.get(OAUTH_CLIENT_ID_ENV, "").strip(),
        OAUTH_CLIENT_SECRET_ENV: os.environ.get(OAUTH_CLIENT_SECRET_ENV, ""),
        OAUTH_REDIRECT_URI_ENV: os.environ.get(OAUTH_REDIRECT_URI_ENV, "").strip(),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"OAuth is enabled but required configuration is missing: {', '.join(missing)}")
    redirects = tuple(
        item.strip()
        for item in required[OAUTH_REDIRECT_URI_ENV].replace("\n", ",").split(",")
        if item.strip()
    )
    return OAuthConfig(
        issuer=required[OAUTH_ISSUER_ENV],
        client_id=required[OAUTH_CLIENT_ID_ENV],
        client_secret=required[OAUTH_CLIENT_SECRET_ENV],
        redirect_uris=redirects,
        scope=os.environ.get(OAUTH_SCOPE_ENV, "hermes").strip() or "hermes",
        pkce_mode=os.environ.get(OAUTH_PKCE_MODE_ENV, "required").strip().lower() or "required",
    )


def static_bearer_from_env() -> str | None:
    token_value = os.environ.get(AUTH_TOKEN_ENV, "")
    if not token_value:
        return None
    if not _CLIENT_SECRET.fullmatch(token_value):
        raise ValueError(f"{AUTH_TOKEN_ENV} must contain 43 to 128 URL-safe characters.")
    return token_value


def _s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _valid_pkce_verifier(verifier: str) -> bool:
    return bool(_PKCE_VALUE.fullmatch(verifier))


def _error_response(exc: OAuthError) -> JSONResponse:
    headers = {"Cache-Control": "no-store", "Pragma": "no-cache"}
    if exc.error == "invalid_client":
        headers["WWW-Authenticate"] = "Basic realm=oauth-token"
    return JSONResponse(
        {"error": exc.error, "error_description": exc.description},
        status_code=exc.status_code,
        headers=headers,
    )


def validate_bearer_token(token_value: str, state: OAuthState | None, *, static_token: str | None = None) -> bool:
    expected = (static_bearer_from_env() or "") if static_token is None else static_token
    if expected and token_value and hmac.compare_digest(token_value, expected):
        return True
    return bool(state and state.validate_access_token(token_value))


class DefaultMcpAcceptMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/mcp":
            headers = list(scope.get("headers") or [])
            accept_indexes = [index for index, (key, _value) in enumerate(headers) if key.lower() == b"accept"]
            replacement = b"application/json, text/event-stream"
            if not accept_indexes:
                headers.append((b"accept", replacement))
                scope = {**scope, "headers": headers}
            else:
                index = accept_indexes[-1]
                value = headers[index][1].decode("latin-1").strip()
                if not value or value == "*/*":
                    headers[index] = (b"accept", replacement)
                    scope = {**scope, "headers": headers}
        await self.app(scope, receive, send)


class BearerAuthMiddleware:
    PUBLIC_PATHS = {
        "/",
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/oauth-authorization-server",
        # ChatGPT probes OIDC discovery after OAuth even when OIDC is disabled.
        # Keep this path unauthenticated so the normal router can return 404
        # (no OpenID Provider is implemented) instead of an auth-challenge.
        "/.well-known/openid-configuration",
        "/oauth/authorize",
        "/oauth/token",
        # RFC 7591 dynamic client registration (public clients, redirect
        # allowlist enforced inside the handler; see register_client).
        "/oauth/register",
    }

    def __init__(self, app: ASGIApp, state: OAuthState | None = None, *, static_token: str | None = None) -> None:
        self.app = app
        self.state = state
        self.static_token = static_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        expected_static = (static_bearer_from_env() or "") if self.static_token is None else self.static_token
        if not expected_static and self.state is None:
            await self.app(scope, receive, send)
            return
        if scope.get("method") == "OPTIONS" or scope.get("path") in self.PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers") or []}
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        supplied = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
        if not validate_bearer_token(supplied, self.state, static_token=expected_static):
            response = JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def authorization_metadata(_request: Request, state: OAuthState) -> JSONResponse:
    issuer = state.config.issuer
    return JSONResponse(
        {
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/oauth/authorize",
            "token_endpoint": f"{issuer}/oauth/token",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic", "none"],
            "code_challenge_methods_supported": ["S256"],
            "scopes_supported": list(state.config.supported_scopes),
        }
    )


def protected_resource_metadata(_request: Request, state: OAuthState) -> JSONResponse:
    return JSONResponse(
        {
            "resource": state.config.resource,
            "authorization_servers": [state.config.issuer],
            "bearer_methods_supported": ["header"],
            "scopes_supported": list(state.config.supported_scopes),
        }
    )


def _redirect_response(redirect_uri: str, values: list[tuple[str, str]]) -> RedirectResponse:
    parsed = urllib.parse.urlparse(redirect_uri)
    existing = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    location = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(existing + values)))
    return RedirectResponse(location, status_code=302)


def _redirect_uri_allowed(redirect_uri: str, config: OAuthConfig) -> bool:
    """Exact match, or prefix match against an entry ending in ``*``."""
    for allowed in config.redirect_uris:
        if allowed.endswith("*"):
            if redirect_uri.startswith(allowed[:-1]):
                return True
        elif redirect_uri == allowed:
            return True
    return False


def authorize(request: Request, state: OAuthState) -> JSONResponse | RedirectResponse:
    params = request.query_params
    client_id = params.get("client_id", "")
    redirect_uri = params.get("redirect_uri", "")
    dynamic = state.is_dynamic_client(client_id)
    if not dynamic and client_id != state.config.client_id:
        return _error_response(OAuthError("invalid_client", "Unknown OAuth client.", status_code=401))
    if dynamic:
        if not state.client_redirect_allowed(client_id, redirect_uri):
            return _error_response(
                OAuthError("invalid_request", "redirect_uri is not registered for this client.")
            )
    elif not _redirect_uri_allowed(redirect_uri, state.config):
        return _error_response(OAuthError("invalid_request", "redirect_uri is not registered."))
    try:
        if params.get("response_type", "") != "code":
            raise OAuthError("unsupported_response_type", "Only response_type=code is supported.")
        scope = state.normalize_scope(params.get("scope", "") or state.config.scope)
        resource = params.get("resource", "") or state.config.resource
        if resource != state.config.resource:
            raise OAuthError("invalid_target", "Requested resource is not supported.")
        challenge = params.get("code_challenge", "")
        method = params.get("code_challenge_method", "")
        # PKCE policy is mode-dependent. "required" (default): every code is
        # bound to an S256 challenge, so a stolen/intercepted code cannot be
        # redeemed without the verifier (RFC 7636; public clients have no
        # secret to authenticate with, the challenge IS the client
        # authentication). "optional": pre-2026-08-31 compat — challenge-less
        # authorize is accepted (the code is redeemed confidentially with the
        # client_secret at the token endpoint, see _authenticate_client), a
        # supplied challenge must still be valid S256, and method-without-
        # challenge is rejected. Dynamic (RFC 7591) clients are ALWAYS public
        # clients: PKCE is mandatory regardless of mode.
        if dynamic or state.config.pkce_mode != "optional":
            if not challenge or method != "S256" or not _PKCE_VALUE.fullmatch(challenge):
                raise OAuthError(
                    "invalid_request", "A valid S256 code_challenge (PKCE) is required."
                )
        else:
            if challenge and (method != "S256" or not _PKCE_VALUE.fullmatch(challenge)):
                raise OAuthError(
                    "invalid_request", "Only a valid S256 PKCE challenge is supported."
                )
            if method and not challenge:
                raise OAuthError(
                    "invalid_request", "code_challenge is required when a method is supplied."
                )
        code = state.issue_authorization_code(
            client_id=client_id,
            redirect_uri=redirect_uri,
            scope=scope,
            resource=resource,
            code_challenge=challenge,
        )
    except OAuthError as exc:
        error_query = [("error", exc.error), ("error_description", exc.description)]
        if params.get("state"):
            error_query.append(("state", params["state"]))
        return _redirect_response(redirect_uri, error_query)
    query = [("code", code)]
    if params.get("state"):
        query.append(("state", params["state"]))
    return _redirect_response(redirect_uri, query)


def _form_value(form: dict[str, list[str]], name: str) -> str:
    values = form.get(name) or []
    return values[0] if values else ""


async def register_client(request: Request, state: OAuthState) -> JSONResponse:
    """RFC 7591 dynamic client registration (public clients only).

    2026-09-06 outage fix: OpenAI's platform calls POST /oauth/register and
    then authorizes with the minted ephemeral client_id — both were 401
    (endpoint absent / unknown client), bricking every headless re-auth.
    Public clients only: no client secrets are ever issued; PKCE and the
    chatgpt.com-family redirect allowlist are the binding controls.
    """
    try:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise OAuthError("invalid_request", "Registration requests must use JSON.")
        content_length = request.headers.get("content-length", "")
        if content_length:
            try:
                if int(content_length) > _MAX_REGISTER_BYTES:
                    raise OAuthError("invalid_request", "Registration request is too large.")
            except ValueError as exc:
                raise OAuthError("invalid_request", "Invalid Content-Length header.") from exc
        buffered = bytearray()
        async for chunk in request.stream():
            if len(buffered) + len(chunk) > _MAX_REGISTER_BYTES:
                raise OAuthError("invalid_request", "Registration request is too large.")
            buffered.extend(chunk)
        try:
            payload = json.loads(bytes(buffered).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OAuthError("invalid_client_metadata", "Registration document is malformed.") from exc
        if not isinstance(payload, dict):
            raise OAuthError("invalid_client_metadata", "Registration document must be a JSON object.")
        auth_method = payload.get("token_endpoint_auth_method", "none")
        if auth_method != "none":
            raise OAuthError(
                "invalid_client_metadata",
                "Only public clients (token_endpoint_auth_method=none) can register.",
            )
        grant_types = payload.get("grant_types", ["authorization_code", "refresh_token"])
        if not isinstance(grant_types, list) or not set(grant_types).issubset(
            {"authorization_code", "refresh_token"}
        ):
            raise OAuthError(
                "invalid_client_metadata",
                "Only authorization_code and refresh_token grants are supported.",
            )
        redirect_uris = payload.get("redirect_uris")
        if not isinstance(redirect_uris, list):
            raise OAuthError("invalid_redirect_uri", "redirect_uris must be a JSON array.")
        client = state.register_dynamic_client(redirect_uris)
        _run_persist_hook(state, "register_client")
        return JSONResponse(client.as_public_dict(), status_code=201)
    except OAuthError as exc:
        return _error_response(exc)


def _client_credentials(request: Request, form: dict[str, list[str]]) -> tuple[str, str]:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(authorization.split(" ", 1)[1], validate=True).decode("utf-8")
        except Exception as exc:
            raise OAuthError("invalid_client", "Invalid OAuth client credentials.", status_code=401) from exc
        client_id, separator, client_secret = decoded.partition(":")
        if not separator:
            raise OAuthError("invalid_client", "Invalid OAuth client credentials.", status_code=401)
        return urllib.parse.unquote(client_id), urllib.parse.unquote(client_secret)
    return _form_value(form, "client_id"), _form_value(form, "client_secret")


def _authenticate_client(request: Request, form: dict[str, list[str]], state: OAuthState) -> str:
    client_id, client_secret = _client_credentials(request, form)
    # B4: non-ASCII credentials used to raise TypeError inside
    # hmac.compare_digest and surface as a 500 on /oauth/token. Malformed
    # credentials are a client error — reject them before any comparison.
    for credential in (client_id, client_secret):
        if credential and not credential.isascii():
            raise OAuthError("invalid_client", "Invalid OAuth client credentials.", status_code=400)
    grant_type = _form_value(form, "grant_type")
    # RFC 7591 dynamic clients are public: authenticated by PKCE (code
    # grants) or prior proof-of-possession (refresh grants). No secret path
    # exists for them — a supplied secret never authenticates them.
    if state.is_dynamic_client(client_id):
        if client_secret:
            raise OAuthError("invalid_client", "Invalid OAuth client credentials.", status_code=401)
        if grant_type == "refresh_token":
            return client_id
        if grant_type == "authorization_code" and _valid_pkce_verifier(
            _form_value(form, "code_verifier")
        ):
            return client_id
        raise OAuthError("invalid_client", "Invalid OAuth client credentials.", status_code=401)
    if not hmac.compare_digest(
        client_id.encode("utf-8"), state.config.client_id.encode("utf-8")
    ):
        raise OAuthError("invalid_client", "Invalid OAuth client credentials.", status_code=401)
    # Public PKCE clients (e.g. ChatGPT connectors) send no client_secret.
    # Secretless auth is only accepted for authorization_code grants that
    # carry a syntactically valid PKCE verifier — and that verifier is then
    # REQUIRED to match the S256 challenge stored in the authorization code
    # (see exchange_authorization_code), so the absence of a client_secret
    # never bypasses client authentication. Refresh-token grants stay
    # secretless; refresh tokens are only issued to clients that already
    # completed a verified PKCE exchange.
    if not client_secret and (
        grant_type == "refresh_token"
        or (
            grant_type == "authorization_code"
            and _valid_pkce_verifier(_form_value(form, "code_verifier"))
        )
    ):
        return client_id
    if not hmac.compare_digest(
        client_secret.encode("utf-8"), state.config.client_secret.encode("utf-8")
    ):
        raise OAuthError("invalid_client", "Invalid OAuth client credentials.", status_code=401)
    return client_id


async def token(request: Request, state: OAuthState) -> JSONResponse:
    try:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/x-www-form-urlencoded":
            raise OAuthError("invalid_request", "Token requests must use form encoding.")
        content_length = request.headers.get("content-length", "")
        if content_length:
            try:
                if int(content_length) > MAX_TOKEN_REQUEST_BYTES:
                    raise OAuthError("invalid_request", "Token request is too large.")
            except ValueError as exc:
                raise OAuthError("invalid_request", "Invalid Content-Length header.") from exc
        buffered = bytearray()
        async for chunk in request.stream():
            if len(buffered) + len(chunk) > MAX_TOKEN_REQUEST_BYTES:
                raise OAuthError("invalid_request", "Token request is too large.")
            buffered.extend(chunk)
        body = bytes(buffered).decode("utf-8")
        try:
            form = urllib.parse.parse_qs(body, keep_blank_values=True, max_num_fields=32)
        except ValueError as exc:
            raise OAuthError("invalid_request", "Token request form is invalid.") from exc
        grant_type = _form_value(form, "grant_type")
        if not grant_type:
            raise OAuthError("invalid_request", "grant_type is required.")
        if grant_type not in {"authorization_code", "refresh_token"}:
            raise OAuthError("unsupported_grant_type", "The requested grant type is not supported.")
        client_id = _authenticate_client(request, form, state)
        if grant_type == "authorization_code":
            response = state.exchange_authorization_code(
                code=_form_value(form, "code"),
                client_id=client_id,
                redirect_uri=_form_value(form, "redirect_uri"),
                code_verifier=_form_value(form, "code_verifier"),
                confidential=bool(_form_value(form, "client_secret")),
            )
        else:
            refresh_token = _form_value(form, "refresh_token")
            if not refresh_token:
                raise OAuthError("invalid_request", "refresh_token is required.")
            response = state.exchange_refresh_token(
                refresh_token=refresh_token,
                client_id=client_id,
                requested_scope=_form_value(form, "scope"),
            )
        return JSONResponse(response, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})
    except UnicodeDecodeError:
        return _error_response(OAuthError("invalid_request", "Token request is malformed."))
    except OAuthError as exc:
        return _error_response(exc)
