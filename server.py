"""
MCP server for ChurchSuite (Core API v2, spec 2.112.3).

Modules (8): attendance, addressbook, calendar, children, network, planning, rotas, smallgroups.
Every GET, POST, PUT and DELETE endpoint in those modules is exposed as a tool, named
<module>_<list|get|create|edit|delete>_<resource>. This server CAN CHANGE AND DELETE DATA.

Run:  python server.py        (serves streamable HTTP at http://HOST:PORT/mcp)

Authentication: every person signs in with their OWN ChurchSuite login (OAuth 2.0
authorization-code flow). The server acts as an OAuth proxy between Claude and ChurchSuite:
Claude sends the person to ChurchSuite's login page, ChurchSuite sends them back to this
server, and every API call is then made with that person's token. ChurchSuite therefore
applies their own permissions, and changes show under their own name.

Setup:
  1. In ChurchSuite, register an OAuth app (authorization-code / "web" app). Set its redirect
     URI to  <PUBLIC_BASE_URL>/auth/callback  and note the client ID and secret.
  2. Set the environment variables below and run the server behind HTTPS.
  3. Add the connector in Claude using  <PUBLIC_BASE_URL>/mcp . Each person who connects it
     is asked to sign in to ChurchSuite.

Server config (env):
      PUBLIC_BASE_URL            REQUIRED. The public https address of this server, e.g.
                                 https://churchsuite-mcp.example.org (no trailing /mcp)
      CHURCHSUITE_CLIENT_ID      REQUIRED. OAuth app client ID (from ChurchSuite)
      CHURCHSUITE_CLIENT_SECRET  REQUIRED. OAuth app client secret (keep secret)
      JWT_SIGNING_KEY            optional. Key for the tokens this server issues to Claude.
                                 Defaults to one derived from the client secret.
      FASTMCP_HOME               optional. Where encrypted login sessions are stored. Put this on
                                 persistent disk, or everyone must sign in again after a redeploy.
      CHURCHSUITE_TOKEN_AUTH_METHOD  optional. "client_secret_basic" (default) or
                                 "client_secret_post", if ChurchSuite rejects the default.
      CHURCHSUITE_MODULES        comma-separated subset to enable (default: all eight)
      CHURCHSUITE_READ_ONLY      "true" hides every POST/PUT/DELETE tool and requests only
                                 the .read scopes (default: false)
      CHURCHSUITE_SCOPE          override the requested scopes
      CHURCHSUITE_TIMEZONE       account timezone for calendar times (default Pacific/Auckland)
      CHURCHSUITE_API_BASE       (default https://api.churchsuite.com/v2)
      CHURCHSUITE_AUTHORIZE_URL  (default https://login.churchsuite.com/oauth2/authorize)
      CHURCHSUITE_TOKEN_URL      (default https://login.churchsuite.com/oauth2/token)
      HOST / PORT                bind address for HTTP mode (default 0.0.0.0 / 8000)

Always serve this over HTTPS: the secret travels in the Authorization header.
"""

import hashlib
import os
import time
from datetime import datetime
from typing import Annotated, Any, Literal

import httpx
from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken, OAuthProxy, TokenVerifier
from fastmcp.server.dependencies import get_access_token
from pydantic import Field
from zoneinfo import ZoneInfo

ALL_MODULES = [
    "attendance", "addressbook", "calendar", "children",
    "network", "planning", "rotas", "smallgroups",
]

_raw = os.environ.get("CHURCHSUITE_MODULES", "")
ENABLED = [m.strip().lower() for m in _raw.split(",") if m.strip()] or ALL_MODULES
_unknown = [m for m in ENABLED if m not in ALL_MODULES]
if _unknown:
    raise SystemExit(f"Unknown CHURCHSUITE_MODULES {_unknown}. Available: {', '.join(ALL_MODULES)}")

READ_ONLY = os.environ.get("CHURCHSUITE_READ_ONLY", "").strip().lower() in ("1", "true", "yes")
API_BASE = os.environ.get("CHURCHSUITE_API_BASE", "https://api.churchsuite.com/v2").rstrip("/")
TOKEN_URL = os.environ.get("CHURCHSUITE_TOKEN_URL", "https://login.churchsuite.com/oauth2/token")
AUTHORIZE_URL = os.environ.get("CHURCHSUITE_AUTHORIZE_URL", "https://login.churchsuite.com/oauth2/authorize")
# ChurchSuite reads the clock time of calendar event starts_at/ends_at as the ACCOUNT'S local time and
# ignores any "Z" or offset, but returns true UTC. Tested 19-Oct-2026: sending 2026-10-20T10:00:00Z gave
# an event at 10:00 NZ time (returned as 2026-10-19T21:00:00Z). Set this to the account's timezone.
ACCOUNT_TZ = ZoneInfo(os.environ.get("CHURCHSUITE_TIMEZONE", "Pacific/Auckland"))


def _default_scope() -> str:
    parts = []
    for m in ENABLED:
        parts.append(f"{m}.read")
        if not READ_ONLY and m != "attendance":  # attendance has no write endpoints
            parts.append(f"{m}.write")
    return " ".join(parts)


SCOPE = os.environ.get("CHURCHSUITE_SCOPE") or _default_scope()
if "user" not in SCOPE.split():
    SCOPE += " user"  # needed to check who is signed in (GET /account/users/current)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} must be set. See the setup notes at the top of server.py.")
    return value


class ChurchSuiteTokenVerifier(TokenVerifier):
    """Checks a person's ChurchSuite token by asking ChurchSuite who it belongs to.

    Results are cached briefly so we do not call ChurchSuite on every request.
    """

    CACHE_SECONDS = 300

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[str, tuple[AccessToken, float]] = {}

    async def verify_token(self, token: str) -> AccessToken | None:
        key = hashlib.sha256(token.encode()).hexdigest()
        now = time.time()
        cached = self._cache.get(key)
        if cached and cached[1] > now:
            return cached[0]
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{API_BASE}/account/users/current",
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                )
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            self._cache.pop(key, None)
            return None
        user = r.json().get("data", {})
        access = AccessToken(
            token=token,
            client_id=str(user.get("username") or user.get("id") or "churchsuite-user"),
            scopes=SCOPE.split(),
            subject=str(user.get("id")) if user.get("id") is not None else None,
            claims={"username": user.get("username"), "name": user.get("name"), "email": user.get("email")},
        )
        for k in [k for k, (_, exp) in self._cache.items() if exp <= now]:
            del self._cache[k]
        self._cache[key] = (access, now + self.CACHE_SECONDS)
        return access


auth = OAuthProxy(
    upstream_authorization_endpoint=AUTHORIZE_URL,
    upstream_token_endpoint=TOKEN_URL,
    upstream_client_id=_required_env("CHURCHSUITE_CLIENT_ID"),
    upstream_client_secret=_required_env("CHURCHSUITE_CLIENT_SECRET"),
    token_verifier=ChurchSuiteTokenVerifier(),
    base_url=_required_env("PUBLIC_BASE_URL").rstrip("/"),
    valid_scopes=SCOPE.split(),
    extra_authorize_params={"scope": SCOPE},  # always ask ChurchSuite for exactly these scopes
    jwt_signing_key=os.environ.get("JWT_SIGNING_KEY") or None,
    token_endpoint_auth_method=os.environ.get("CHURCHSUITE_TOKEN_AUTH_METHOD") or None,
)
ALLOWED_PREFIXES = tuple(f"/{m}/" for m in ENABLED)

mcp = FastMCP(
    "churchsuite",
    auth=auth,
    instructions=(
        f"ChurchSuite access, acting as the signed-in person with their own ChurchSuite permissions. Enabled modules: {', '.join(ENABLED)}. "
        + ("READ-ONLY mode: no create/edit/delete tools exist. " if READ_ONLY else
           "Create/edit/delete tools change real data: confirm with the user before using them, "
           "and never delete or overwrite anything the user did not explicitly ask for. ")
        + "List tools are paginated: pass page / per_page (1-250, default 50) and check "
        "pagination.next_page to continue. Array filters match ANY of the given values. "
        "Edit tools send only the fields you pass; to clear a nullable field, name it in "
        "set_null. Attendance: a Gathering has Metricsets containing Metrics; a Record is one "
        "Gathering on one date with Record Metrics (values) and, if people tracking is on, "
        "Record Contacts. Currency values are in the smallest unit (pence/cents). The children "
        "module holds children's personal data: handle it with care. Create/edit tools list, "
        "per field, what to provide and where to find IDs (e.g. category_id: run calendar_list_categories)."
    ),
)

_ANNOTATIONS = {
    "GET": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    "POST": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    "PUT": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    "DELETE": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True},
}


def tool(module: str, method: str):
    """Register the function as an MCP tool only if its module (and write access) is enabled."""

    def deco(fn):
        if module not in ENABLED or (READ_ONLY and method != "GET"):
            return fn
        return mcp.tool(annotations=_ANNOTATIONS[method])(fn)

    return deco


def _auth_header() -> dict[str, str]:
    """Headers for a ChurchSuite API call, using the signed-in person's own token."""
    access = get_access_token()
    if access is None or not access.token:
        raise RuntimeError("Not signed in to ChurchSuite. Reconnect the connector and sign in again.")
    return {"Authorization": f"Bearer {access.token}", "Accept": "application/json"}


async def _request(
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    if method not in ("GET", "POST", "PUT", "DELETE"):
        raise ValueError(f"Method not allowed: {method}")
    if method != "GET" and READ_ONLY:
        raise ValueError("Server is in read-only mode")
    if not path.startswith(ALLOWED_PREFIXES):
        raise ValueError(f"Path not allowed: {path}")
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    async with httpx.AsyncClient(timeout=30) as client:
        headers = _auth_header()
        r = await client.request(
            method, f"{API_BASE}{path}", params=clean, headers=headers,
            **({"json": body} if body is not None else {}),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"ChurchSuite API {r.status_code}: {r.text[:500]}")
        if r.status_code == 204 or not r.text.strip():
            return {"status": r.status_code, "ok": True}
        return r.json()


def _plain(v: Any) -> Any:
    """Plain JSON-able copy of a value (lists/dicts are copied recursively)."""
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def _to_account_local(value: str | None) -> str | None:
    """Convert a date-time to the form ChurchSuite's calendar expects: the account's LOCAL clock time.

    - With a Z or offset (e.g. 2026-11-01T10:00:00Z, or a value read back from the API): converted to
      account-local time.
    - With no offset (e.g. 2026-11-01T19:30:00): taken as already account-local.
    The result is sent with a trailing Z, the format ChurchSuite accepts, but the clock time is local.
    """
    if value is None:
        return None
    v = value.strip()
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00") if v.endswith("Z") else v)
    except ValueError:
        raise ValueError(f"Not a valid ISO 8601 date-time: {value!r} (e.g. 2026-11-01T19:30:00)")
    if dt.tzinfo is not None:
        dt = dt.astimezone(ACCOUNT_TZ).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _body(
    values: dict[str, Any],
    set_null: list[str] | None,
    nullable: tuple[str, ...],
    keep: tuple[str, ...] = (),
) -> dict[str, Any]:
    # `keep` = required fields that may legitimately be null: always sent, even when None.
    body = {k: _plain(v) for k, v in values.items() if v is not None or k in keep}
    for k in set_null or []:
        if k not in nullable:
            raise ValueError(f"'{k}' cannot be set to null. Nullable fields: {', '.join(nullable) or 'none'}")
        body[k] = None
    return body


def _check_choice(body: dict[str, Any], groups: list, need_one: bool) -> None:
    """Enforce 'pass exactly one of these option groups' rules from the spec's oneOf blocks."""
    names = " OR ".join(" + ".join(props) for props, _ in groups)
    used = [(props, req) for props, req in groups if any(body.get(p) is not None for p in props)]
    if len(used) > 1:
        raise ValueError(f"Pass only one of: {names}")
    if not used:
        if need_one:
            raise ValueError(f"Pass one of: {names}")
        return
    missing = [r for r in used[0][1] if body.get(r) is None]
    if missing:
        raise ValueError(f"Missing {', '.join(missing)} (needed with {' + '.join(used[0][0])})")


# ======== attendance ========

@tool("attendance", "GET")
async def attendance_get_gathering(id: int) -> Any:
    """Get a gathering. Retrieve a single gathering identified by ID."""
    return await _request("GET", f"/attendance/gatherings/{id}", None)

@tool("attendance", "GET")
async def attendance_get_metricset(id: int) -> Any:
    """Get a metricset. Retrieve a single metricset identified by ID."""
    return await _request("GET", f"/attendance/metricsets/{id}", None)

@tool("attendance", "GET")
async def attendance_get_record(id: int) -> Any:
    """Get a record. Retrieve a single record identified by ID."""
    return await _request("GET", f"/attendance/records/{id}", None)

@tool("attendance", "GET")
async def attendance_list_gatherings(page: int | None = None, per_page: int | None = None, site_ids: list[int] | None = None, status: Literal["active", "archived"] | None = None, q: str | None = None) -> Any:
    """List gatherings. Returns a list of gatherings, sorted alphabetically by name.
    site_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    status: Filter records by the provided status.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/attendance/gatherings", {"page": page, "per_page": per_page, "site_ids[]": site_ids, "status": status, "q": q})

@tool("attendance", "GET")
async def attendance_list_metrics(page: int | None = None, per_page: int | None = None, metricset_ids: list[int] | None = None) -> Any:
    """List metrics. Returns a list of metrics, sorted alphabetically by order.
    metricset_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/attendance/metrics", {"page": page, "per_page": per_page, "metricset_ids[]": metricset_ids})

@tool("attendance", "GET")
async def attendance_list_metricsets(page: int | None = None, per_page: int | None = None, gathering_ids: list[int] | None = None) -> Any:
    """List metricsets. Returns a list of metricsets, sorted alphabetically by order.
    gathering_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/attendance/metricsets", {"page": page, "per_page": per_page, "gathering_ids[]": gathering_ids})

@tool("attendance", "GET")
async def attendance_list_record_contacts(page: int | None = None, per_page: int | None = None, contact_ids: list[int] | None = None, record_ids: list[int] | None = None) -> Any:
    """List record contacts. Returns a list of contacts who were recorded as attending a gathering.
    contact_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    record_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/attendance/record_contacts", {"page": page, "per_page": per_page, "contact_ids[]": contact_ids, "record_ids[]": record_ids})

@tool("attendance", "GET")
async def attendance_list_record_metrics(page: int | None = None, per_page: int | None = None, metric_ids: list[int] | None = None, record_ids: list[int] | None = None) -> Any:
    """List record metrics. Returns a list of record metrics, i.e. number/currency values that have been recorded for an attendance record.
    metric_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    record_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/attendance/record_metrics", {"page": page, "per_page": per_page, "metric_ids[]": metric_ids, "record_ids[]": record_ids})

@tool("attendance", "GET")
async def attendance_list_records(page: int | None = None, per_page: int | None = None, days: list[Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]] | None = None, gathering_ids: list[int] | None = None, is_before: str | None = None, is_after: str | None = None) -> Any:
    """List records. Returns a list of records, sorted by date.
    days: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    gathering_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    is_before: Only include records that are before this date.
    is_after: Only include records that are after this date."""
    return await _request("GET", "/attendance/records", {"page": page, "per_page": per_page, "days[]": days, "gathering_ids[]": gathering_ids, "is_before": is_before, "is_after": is_after})



# ======== addressbook ========

@tool("addressbook", "GET")
async def addressbook_get_contact(id: int) -> Any:
    """Get a contact. Retrieve a single contact identified by ID."""
    return await _request("GET", f"/addressbook/contacts/{id}", None)

@tool("addressbook", "GET")
async def addressbook_get_contact_note(id: int) -> Any:
    """List contact notes. Returns a list of contact notes, sorted by pinned, created_at."""
    return await _request("GET", f"/addressbook/contacts/{id}/notes", None)

@tool("addressbook", "GET")
async def addressbook_get_custom_field(id: int) -> Any:
    """Get a custom field. Retrieve a single custom field identified by ID."""
    return await _request("GET", f"/addressbook/custom_fields/{id}", None)

@tool("addressbook", "GET")
async def addressbook_get_flow(id: int) -> Any:
    """Get a flow. Retrieve a single flow identified by ID."""
    return await _request("GET", f"/addressbook/flows/{id}", None)

@tool("addressbook", "GET")
async def addressbook_get_flow_stage(id: int) -> Any:
    """Get a flow stage. Retrieve a single stage identified by ID."""
    return await _request("GET", f"/addressbook/flow_stages/{id}", None)

@tool("addressbook", "GET")
async def addressbook_get_flow_tracking(id: int) -> Any:
    """Get a flow tracking. Retrieve a single flow tracking, linked to a person's process through a flow, identified by ID."""
    return await _request("GET", f"/addressbook/flow_trackings/{id}", None)

@tool("addressbook", "GET")
async def addressbook_get_form_field(id: int) -> Any:
    """Get a form field. Retrieve a single form field identified by ID."""
    return await _request("GET", f"/addressbook/form_fields/{id}", None)

@tool("addressbook", "GET")
async def addressbook_get_form_response(id: int) -> Any:
    """Get a form response. Retrieve a single form response identified by ID."""
    return await _request("GET", f"/addressbook/form_responses/{id}", None)

@tool("addressbook", "GET")
async def addressbook_get_key_date(id: int) -> Any:
    """Get a key date. Retrieve a single key date identified by ID."""
    return await _request("GET", f"/addressbook/key_dates/{id}", None)

@tool("addressbook", "GET")
async def addressbook_get_tag(id: int) -> Any:
    """Get a tag. Retrieve a single tag identified by ID."""
    return await _request("GET", f"/addressbook/tags/{id}", None)

@tool("addressbook", "GET")
async def addressbook_get_tag_category(id: int) -> Any:
    """Get a tag category. Retrieve a single tag category identified by ID"""
    return await _request("GET", f"/addressbook/tag_categories/{id}", None)

@tool("addressbook", "GET")
async def addressbook_list_contacts(page: int | None = None, per_page: int | None = None, site_ids: list[int] | None = None, status: Literal["active", "archived", "pending"] | None = None, q: str | None = None, tag_ids: list[int] | None = None, name_visible: bool | None = None, address_visible: bool | None = None, mobile_visible: bool | None = None, email_visible: bool | None = None, telephone_visible: bool | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List contacts. Returns a list of contacts.
    site_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    status: Filter records by the provided status.
    q: Filter records with a fuzzy string match against the following properties: first_name, last_name, formal_name, email, telephone, mobile, address, job, employer.
    tag_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    name_visible: Filter records to those matching marked as having their name visible in My ChurchSuite.
    address_visible: Filter records to those matching marked as having their address visible in My ChurchSuite.
    mobile_visible: Filter records to those matching marked as having their mobile visible in My ChurchSuite.
    email_visible: Filter records to those matching marked as having their email visible in My ChurchSuite.
    telephone_visible: Filter records to those matching marked as having their telephone visible in My ChurchSuite.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/addressbook/contacts", {"page": page, "per_page": per_page, "site_ids[]": site_ids, "status": status, "q": q, "tag_ids[]": tag_ids, "name_visible": name_visible, "address_visible": address_visible, "mobile_visible": mobile_visible, "email_visible": email_visible, "telephone_visible": telephone_visible, "order_by": order_by})

@tool("addressbook", "GET")
async def addressbook_list_custom_fields(page: int | None = None, per_page: int | None = None) -> Any:
    """List custom fields. Returns a list of custom fields"""
    return await _request("GET", "/addressbook/custom_fields", {"page": page, "per_page": per_page})

@tool("addressbook", "GET")
async def addressbook_list_flow_stages(page: int | None = None, per_page: int | None = None, flow_ids: list[int] | None = None) -> Any:
    """List flow stages. Returns a list of stages.
    flow_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/addressbook/flow_stages", {"page": page, "per_page": per_page, "flow_ids[]": flow_ids})

@tool("addressbook", "GET")
async def addressbook_list_flow_trackings(page: int | None = None, per_page: int | None = None, due_before: str | None = None, due_after: str | None = None, flow_ids: list[int] | None = None, contact_ids: list[int] | None = None, stage_ids: list[int] | None = None) -> Any:
    """List flow trackings. Returns a list of flow trackings linked to people's progress through a flow.
    due_before: Only include records that start before this date.
    due_after: Only include records that start after this date.
    flow_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    contact_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    stage_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/addressbook/flow_trackings", {"page": page, "per_page": per_page, "due_before": due_before, "due_after": due_after, "flow_ids[]": flow_ids, "contact_ids[]": contact_ids, "stage_ids[]": stage_ids})

@tool("addressbook", "GET")
async def addressbook_list_flows(page: int | None = None, per_page: int | None = None, status: Literal["active", "archived"] | None = None, q: str | None = None) -> Any:
    """List flows. Returns a list of flows, sorted alphabetically by name.
    status: Filter records by the provided status.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/addressbook/flows", {"page": page, "per_page": per_page, "status": status, "q": q})

@tool("addressbook", "GET")
async def addressbook_list_form_fields(page: int | None = None, per_page: int | None = None, form_ids: list[int] | None = None) -> Any:
    """List form fields. Returns a list of form fields
    form_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/addressbook/form_fields", {"page": page, "per_page": per_page, "form_ids[]": form_ids})

@tool("addressbook", "GET")
async def addressbook_list_form_responses(page: int | None = None, per_page: int | None = None, form_ids: list[int] | None = None, statuses: list[Literal["active", "archived", "spam"]] | None = None, q: str | None = None, has_linked_resource: bool | None = None, invite_response: bool | None = None) -> Any:
    """List form responses. Returns a list of form responses, sorted alphabetically by name.
    form_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    statuses: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    q: Filter records with a fuzzy string match against the following properties: first_name, last_name.
    has_linked_resource: Filter records by the provided status.
    invite_response: Filter records by the provided status."""
    return await _request("GET", "/addressbook/form_responses", {"page": page, "per_page": per_page, "form_ids[]": form_ids, "statuses[]": statuses, "q": q, "has_linked_resource": has_linked_resource, "invite_response": invite_response})

@tool("addressbook", "GET")
async def addressbook_list_key_date_resources(page: int | None = None, per_page: int | None = None, key_date_ids: list[int] | None = None, contact_ids: list[int] | None = None, order_by: Literal["created_at:asc", "created_at:desc", "date:asc", "date:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc"] | None = None) -> Any:
    """List key date resources. Returns a list of key date resources, optionally filtered by key date or contact ID. A key date resource records an instance of a key date for a contact on a given date.
    key_date_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    contact_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/addressbook/key_date_resources", {"page": page, "per_page": per_page, "key_date_ids[]": key_date_ids, "contact_ids[]": contact_ids, "order_by": order_by})

@tool("addressbook", "GET")
async def addressbook_list_key_dates(page: int | None = None, per_page: int | None = None, q: str | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List key dates. Returns a list of key dates, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/addressbook/key_dates", {"page": page, "per_page": per_page, "q": q, "order_by": order_by})

@tool("addressbook", "GET")
async def addressbook_list_tag_categories() -> Any:
    """List tag categories. Returns a list of tag categories, sorted alphabetically by name."""
    return await _request("GET", "/addressbook/tag_categories", None)

@tool("addressbook", "GET")
async def addressbook_list_tag_resources(page: int | None = None, per_page: int | None = None, contact_ids: list[int] | None = None, tag_ids: list[int] | None = None) -> Any:
    """List tag resources. Returns a list of tag resources. A tag resource is the relational model linking a contact to a tag.
    contact_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    tag_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/addressbook/tag_resources", {"page": page, "per_page": per_page, "contact_ids[]": contact_ids, "tag_ids[]": tag_ids})

@tool("addressbook", "GET")
async def addressbook_list_tags(page: int | None = None, per_page: int | None = None, q: str | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List tags. Returns a list of tags, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/addressbook/tags", {"page": page, "per_page": per_page, "q": q, "order_by": order_by})

@tool("addressbook", "POST")
async def addressbook_create_contact(
    first_name: Annotated[str, Field(description="First name (first_name) - REQUIRED. The first name of the Contact. (max 50 characters; cannot be empty)")],
    last_name: Annotated[str, Field(description="Last name (last_name) - REQUIRED. The last name of the Contact. (max 50 characters; cannot be empty)")],
    all_sites: Annotated[bool, Field(description="All sites (all_sites) - REQUIRED. Whether the Contact belongs to All Sites.")],
    site_ids: Annotated[list[int], Field(description="Site IDs (site_ids) - REQUIRED. An array of Site IDs that the Contact belongs to. If `all_sites` is `true`, this will be overridden. You are only able to add site IDs for sites you have the relevant permission for. HOW TO GET IT: No list tool for sites. Copy `site_ids` from an existing record (e.g. addressbook_get_contact), or use all_sites=true for every site.")],
    middle_name: Annotated[str | None, Field(description="Middle name (middle_name) - Optional. The middle name of the Contact. (max 50 characters; cannot be empty)")] = None,
    title: Annotated[str | None, Field(description="Title (title) - Optional. The title of the Contact. (max 25 characters; cannot be empty)")] = None,
    formal_name: Annotated[str | None, Field(description="Formal name (formal_name) - Optional. The formal name of the Contact. (max 50 characters; cannot be empty)")] = None,
    former_name: Annotated[str | None, Field(description="Former name (former_name) - Optional. Use this if a Contact has changed their last name and you want to record their previous last name. I.e. someone's maiden name. (max 50 characters; cannot be empty)")] = None,
    email: Annotated[str | None, Field(description="Email (email) - Optional. The Contact's email address. (max 100 characters; an email address)")] = None,
    mobile: Annotated[str | None, Field(description="Mobile (mobile) - Optional. The Contact's mobile number. (max 20 characters; cannot be empty)")] = None,
    telephone: Annotated[str | None, Field(description="Telephone (telephone) - Optional. The Contact's telephone number. (max 20 characters; cannot be empty)")] = None,
    address: Annotated[dict[str, Any] | None, Field(description="Address (address) - Optional. A street Address. Object keys: {line_1 (string): The first line of the Address; line_2 (string): The second line of the Address; line_3 (string): The third line of the Address; city (string): The city of the Address; county (string): The county of the Address; country (string): The country of the Address, in ISO3166 alpha 2 format; postcode (string): The postcode of the Address}")] = None,
    sex: Annotated[Literal["female", "male"] | None, Field(description="Sex (sex) - Optional. The sex of the Contact. (one of: female, male)")] = None,
    date_of_birth: Annotated[str | None, Field(description="Date of birth (date_of_birth) - Optional. The Contact's date of birth. (date as YYYY-MM-DD)")] = None,
    marital_status: Annotated[Literal["single", "married", "engaged", "partnered", "separated", "divorced", "widowed"] | None, Field(description="Marital status (marital_status) - Optional. The Contact's marital status. (one of: single, married, engaged, partnered, separated, divorced, widowed)")] = None,
    spouse_id: Annotated[int | None, Field(description="Spouse ID (spouse_id) - Optional. The Contact ID for the Contact's spouse. HOW TO GET IT: Run addressbook_list_contacts and use the spouse's contact `id`.")] = None,
    job: Annotated[str | None, Field(description="Job (job) - Optional. The job of the Contact. (max 100 characters; cannot be empty)")] = None,
    employer: Annotated[str | None, Field(description="Employer (employer) - Optional. The employer of the Contact. (max 100 characters; cannot be empty)")] = None,
    work_telephone: Annotated[str | None, Field(description="Work telephone (work_telephone) - Optional. The Contact's work telephone number. (max 20 characters; cannot be empty)")] = None,
    custom_fields: Annotated[list[dict[str, Any]] | None, Field(description="Custom fields (custom_fields) - Optional. Custom Field responses for this resource. HOW TO GET IT: Use addressbook_list_custom_fields to find each field's `id` and type. Object keys: list of ONE of these shapes: {id (integer, required): The ID of the Custom Field; field_type (one of text/textarea/phone/date, required): A discriminator to determine the type of Custom Field this response belongs to; value (string, required): The raw value of a text, textarea, phone or date} OR {id (integer, required): The ID of the Custom Field; field_type (one of email, required): A discriminator to determine the type of Custom Field this response belongs to; value (string, required): The raw value of an email} OR {id (integer, required): The ID of the Custom Field; field_type (one of radio/select, required): A discriminator to determine the type of Custom Field this response belongs to; value ({id (string, required)}, required): The UUID of the selected response for a radio or select field} OR {id (integer, required): The ID of the Custom Field; field_type (one of checkbox, required): A discriminator to determine the type of Custom Field this response belongs to; value (list of {id (string, required)}, required): A list of UUIDs of the selected responses for a checkbox field}")] = None,
    communication: Annotated[dict[str, Any] | None, Field(description="Communication (communication) - Optional. The person's communication preferences. If no data is provided, the module default communication preferences will be used. Object keys: {general_email (boolean, required): Whether this person has consented to receiving general emails; general_sms (boolean, required): Whether this person has consented to receiving general SMS messages; phone (boolean, required): Whether this person has consented to receiving phone calls; post (boolean, required): Whether this person has consented to receiving post; rota_email (boolean, required): Whether this person has consented to receiving rota emails; rota_sms (boolean, required): Whether this person has consented to receiving rota SMS messages}")] = None,
    login: Annotated[dict[str, Any] | None, Field(description="Login (login) - Optional. My ChurchSuite/CharitySuite login information. Object keys: {enabled (boolean, required): Whether the person is able to log into My; invited (boolean, required): Whether the person has been invited to create a My account}")] = None,
    privacy: Annotated[dict[str, Any] | None, Field(description="Privacy (privacy) - Optional. Add the Contact's privacy settings. If no data is provided, the default privacy settings will be used. Object keys: {name_visible (boolean, required): Whether the Contact's name is visible in My; address_visible (boolean, required): Whether the Contact's address is visible in My. This will only be respected if name_visible is true, else it will not be visible; mobile_visible (boolean, required): Whether the Contact's mobile is visible in My. This will only be respected if name_visible is true, else it will not be visible; email_visible (boolean, required): Whether the Contact's email is visible in My. This will only be respected if name_visible is true, else it will not be visible; telephone_visible (boolean, required): Whether the Contact's mobile is visible in My. This will only be respected if name_visible is true, else it will not be visible}")] = None,
) -> Any:
    """Create a contact. Create a contact with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - first_name (required): The first name of the Contact (max 50 characters; cannot be empty)
      - last_name (required): The last name of the Contact (max 50 characters; cannot be empty)
      - all_sites (required): Whether the Contact belongs to All Sites
      - site_ids (required): An array of Site IDs that the Contact belongs to. If `all_sites` is `true`, this will be overridden. You are only able to add site IDs for sites you have the relevant permission for -> No list tool for sites. Copy `site_ids` from an existing record (e.g. addressbook_get_contact), or use all_sites=true for every site.
      - middle_name (optional): The middle name of the Contact (max 50 characters; cannot be empty)
      - title (optional): The title of the Contact (max 25 characters; cannot be empty)
      - formal_name (optional): The formal name of the Contact (max 50 characters; cannot be empty)
      - former_name (optional): Use this if a Contact has changed their last name and you want to record their previous last name. I.e. someone's maiden name (max 50 characters; cannot be empty)
      - email (optional): The Contact's email address (max 100 characters; an email address)
      - mobile (optional): The Contact's mobile number (max 20 characters; cannot be empty)
      - telephone (optional): The Contact's telephone number (max 20 characters; cannot be empty)
      - address (optional): A street Address
      - sex (optional): The sex of the Contact (one of: female, male)
      - date_of_birth (optional): The Contact's date of birth (date as YYYY-MM-DD)
      - marital_status (optional): The Contact's marital status (one of: single, married, engaged, partnered, separated, divorced, widowed)
      - spouse_id (optional): The Contact ID for the Contact's spouse -> Run addressbook_list_contacts and use the spouse's contact `id`.
      - job (optional): The job of the Contact (max 100 characters; cannot be empty)
      - employer (optional): The employer of the Contact (max 100 characters; cannot be empty)
      - work_telephone (optional): The Contact's work telephone number (max 20 characters; cannot be empty)
      - custom_fields (optional): Custom Field responses for this resource -> Use addressbook_list_custom_fields to find each field's `id` and type.
      - communication (optional): The person's communication preferences. If no data is provided, the module default communication preferences will be used
      - login (optional): My ChurchSuite/CharitySuite login information
      - privacy (optional): Add the Contact's privacy settings. If no data is provided, the default privacy settings will be used"""
    body = _body({"first_name": first_name, "last_name": last_name, "middle_name": middle_name, "title": title, "formal_name": formal_name, "former_name": former_name, "email": email, "mobile": mobile, "telephone": telephone, "address": address, "sex": sex, "date_of_birth": date_of_birth, "marital_status": marital_status, "spouse_id": spouse_id, "job": job, "employer": employer, "work_telephone": work_telephone, "custom_fields": custom_fields, "communication": communication, "login": login, "privacy": privacy, "all_sites": all_sites, "site_ids": site_ids}, None, (), keep=())
    return await _request("POST", "/addressbook/contacts", None, body)

@tool("addressbook", "POST")
async def addressbook_create_contact_note(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the contact to add the note to. Find it with addressbook_list_contacts (use `q` to search).")],
    content: Annotated[str, Field(description="Content (content) - REQUIRED. The content of the Note. (max 65535 characters; cannot be empty)")],
    pinned: Annotated[bool | None, Field(description="Pinned (pinned) - Optional. Whether the Note should be pinned.")] = None,
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - Optional. Object keys: ONE of these shapes: {just_me (one of true): The resource should only be visible to the current user} OR {user_group_ids (list of integer): List of user group IDs the resource is visible to. You are only able to add user group IDs for user groups you have access to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")] = None,
) -> Any:
    """Create a contact note. Create a Note against a Contact in the Address Book module.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the contact to add the note to. Find it with addressbook_list_contacts (use `q` to search).
      - content (required): The content of the Note (max 65535 characters; cannot be empty)
      - pinned (optional): Whether the Note should be pinned
      - visible_to (optional): Visible to"""
    body = _body({"content": content, "pinned": pinned, "visible_to": visible_to}, None, (), keep=())
    return await _request("POST", f"/addressbook/contacts/{id}/notes", None, body)

@tool("addressbook", "POST")
async def addressbook_create_flow_stage(
    flow_id: Annotated[int, Field(description="Flow ID (flow_id) - REQUIRED. The Flow ID the Stage belongs to. HOW TO GET IT: Run addressbook_list_flows and use the flow `id`.")],
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Stage's name. (max 50 characters; cannot be empty)")],
    due_interval: Annotated[int, Field(description="Due interval (due_interval) - REQUIRED. The number of days, weeks or months used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage.")],
    due_period: Annotated[Literal["days", "weeks", "months"], Field(description="Due period (due_period) - REQUIRED. The due period used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage. (one of: days, weeks, months)")],
    due_from: Annotated[Literal["first_added", "previous_stage"], Field(description="Due from (due_from) - REQUIRED. The relative point used for tracking a person linked to the Stage, either when they are first added to the flow or when they are processed from a previous stage. (one of: first_added, previous_stage)")],
    instructions: Annotated[str | None, Field(description="Instructions (instructions) - Optional. The instructions for the Stage. (max 65535 characters)")] = None,
    assigned_user_id: Annotated[int | None, Field(description="Assigned user ID (assigned_user_id) - Optional. The ID of the user automatically assigned to people in this stage, or null if no user is assigned. HOW TO GET IT: A ChurchSuite user ID. No list tool; it appears as `assigned_user_id` on existing flow stages/trackings.")] = None,
    automate_time: Annotated[str | None, Field(description="Automate time (automate_time) - Optional. The UTC time the Stage will automatically process any due or overdue people if it is automated, or null if it is not automated.")] = None,
) -> Any:
    """Create a flow stage. Create a stage for a flow.
    
    WHAT YOU NEED TO PROVIDE
      - flow_id (required): The Flow ID the Stage belongs to -> Run addressbook_list_flows and use the flow `id`.
      - name (required): The Stage's name (max 50 characters; cannot be empty)
      - due_interval (required): The number of days, weeks or months used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage
      - due_period (required): The due period used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage (one of: days, weeks, months)
      - due_from (required): The relative point used for tracking a person linked to the Stage, either when they are first added to the flow or when they are processed from a previous stage (one of: first_added, previous_stage)
      - instructions (optional): The instructions for the Stage (max 65535 characters)
      - assigned_user_id (optional): The ID of the user automatically assigned to people in this stage, or null if no user is assigned -> A ChurchSuite user ID. No list tool; it appears as `assigned_user_id` on existing flow stages/trackings.
      - automate_time (optional): The UTC time the Stage will automatically process any due or overdue people if it is automated, or null if it is not automated"""
    body = _body({"flow_id": flow_id, "name": name, "instructions": instructions, "due_interval": due_interval, "due_period": due_period, "due_from": due_from, "assigned_user_id": assigned_user_id, "automate_time": automate_time}, None, (), keep=())
    return await _request("POST", "/addressbook/flow_stages", None, body)

@tool("addressbook", "POST")
async def addressbook_create_flow_tracking(
    flow_id: Annotated[int, Field(description="Flow ID (flow_id) - REQUIRED. The Flow ID the Tracking is linked to. HOW TO GET IT: Run addressbook_list_flows and use the flow `id`.")],
    stage_id: Annotated[int, Field(description="Stage ID (stage_id) - REQUIRED. The Stage ID the Tracking is linked to. HOW TO GET IT: Run addressbook_list_flow_stages (filter flow_ids) and use the stage `id`.")],
    person: Annotated[dict[str, Any], Field(description="Person (person) - REQUIRED. The Tracking's Person details. Object keys: {type (string, required): The Person's type, indicating whether they are a Child, Contact, Customer or Giver -> For a contact use 'addressbook_contact'.; id (integer, required): The Person's ID -> Contact ID: run addressbook_list_contacts (search with `q`) and use the `id`.}")],
) -> Any:
    """Create a flow tracking. Create a flow tracking, linked to a person's progress through a flow, with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - flow_id (required): The Flow ID the Tracking is linked to -> Run addressbook_list_flows and use the flow `id`.
      - stage_id (required): The Stage ID the Tracking is linked to -> Run addressbook_list_flow_stages (filter flow_ids) and use the stage `id`.
      - person (required): The Tracking's Person details"""
    body = _body({"flow_id": flow_id, "stage_id": stage_id, "person": person}, None, (), keep=())
    return await _request("POST", "/addressbook/flow_trackings", None, body)

@tool("addressbook", "POST")
async def addressbook_create_key_date(
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Key Date's name. (max 50 characters; cannot be empty)")],
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], Field(description="Colour (colour) - REQUIRED. The Key Date's colour. (one of: blue, red, orange, green, purple, pink, brown, grey)")],
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - Optional. Object keys: ONE of these shapes: {just_me (one of true): The resource should only be visible to the current user} OR {user_group_ids (list of integer): List of user group IDs the resource is visible to. You are only able to add user group IDs for user groups you have access to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")] = None,
) -> Any:
    """Create a key date. Create a key date with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - name (required): The Key Date's name (max 50 characters; cannot be empty)
      - colour (required): The Key Date's colour (one of: blue, red, orange, green, purple, pink, brown, grey)
      - visible_to (optional): Visible to"""
    body = _body({"name": name, "colour": colour, "visible_to": visible_to}, None, (), keep=())
    return await _request("POST", "/addressbook/key_dates", None, body)

@tool("addressbook", "POST")
async def addressbook_create_key_date_resource(
    key_date_id: Annotated[int, Field(description="Key date ID (key_date_id) - REQUIRED. The Key Date's ID. HOW TO GET IT: Run addressbook_list_key_dates and use the key date `id`.")],
    resource: Annotated[dict[str, Any], Field(description="Resource (resource) - REQUIRED. The resource to link to the Key Date. Object keys: {id (integer, required): The ID of the resource -> Contact ID: addressbook_list_contacts.}")],
    date: Annotated[str, Field(description="Date (date) - REQUIRED. The date of the Key Date Resource. (date as YYYY-MM-DD)")],
    description: Annotated[str | None, Field(description="Description (description) - Optional. The description of this Key Date Resource. (max 65535 characters)")] = None,
) -> Any:
    """Create a key date resource. Create a key date resource adding a contact to a key date
    
    WHAT YOU NEED TO PROVIDE
      - key_date_id (required): The Key Date's ID -> Run addressbook_list_key_dates and use the key date `id`.
      - resource (required): The resource to link to the Key Date
      - date (required): The date of the Key Date Resource (date as YYYY-MM-DD)
      - description (optional): The description of this Key Date Resource (max 65535 characters)"""
    body = _body({"key_date_id": key_date_id, "resource": resource, "date": date, "description": description}, None, (), keep=())
    return await _request("POST", "/addressbook/key_date_resources", None, body)

@tool("addressbook", "POST")
async def addressbook_create_tag(
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Tag's name. (max 50 characters; cannot be empty)")],
    is_smart: Annotated[bool, Field(description="Is smart (is_smart) - REQUIRED. Whether the Tag is smart or fixed.")],
    category_id: Annotated[int | None, Field(description="Category ID (category_id) - Optional. The Tag's Category ID. This sets the default colour of the Tag to the Category's colour. HOW TO GET IT: Run addressbook_list_tag_categories and use the `id` of the category.")] = None,
    description: Annotated[str | None, Field(description="Description (description) - Optional. The description of the Tag. (max 65535 characters)")] = None,
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"] | None, Field(description="Colour (colour) - Optional. The name of the colour of the Tag. If not specified, the Category colour will be used where applicable, falling back to 'blue'. (one of: blue, red, orange, green, purple, pink, brown, grey)")] = None,
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - Optional. Object keys: ONE of these shapes: {just_me (one of true): The resource should only be visible to the current user} OR {user_group_ids (list of integer): List of user group IDs the resource is visible to. You are only able to add user group IDs for user groups you have access to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")] = None,
    pinned_icon: Annotated[Literal["tick", "cross", "archive", "warning", "flag", "magic", "star", "profile", "user", "couple", "group", "pledge", "education", "job", "gift", "role", "duration", "smiley-happy", "smiley-neutral", "smiley-sad"] | None, Field(description="Pinned icon (pinned_icon) - Optional. Pin this Tag by specifying an icon. Pinned tags give you a quick way to see your most important tags at a glance; they're displayed at the top of the page when viewing a person's profile. (one of: tick, cross, archive, warning, flag, magic, star, profile, user, couple, group, pledge, education, job, gift, role, duration, smiley-happy, smiley-neutral, smiley-sad)")] = None,
) -> Any:
    """Create a tag.
    
    WHAT YOU NEED TO PROVIDE
      - name (required): The Tag's name (max 50 characters; cannot be empty)
      - is_smart (required): Whether the Tag is smart or fixed
      - category_id (optional): The Tag's Category ID. This sets the default colour of the Tag to the Category's colour -> Run addressbook_list_tag_categories and use the `id` of the category.
      - description (optional): The description of the Tag (max 65535 characters)
      - colour (optional): The name of the colour of the Tag. If not specified, the Category colour will be used where applicable, falling back to 'blue' (one of: blue, red, orange, green, purple, pink, brown, grey)
      - visible_to (optional): Visible to
      - pinned_icon (optional): Pin this Tag by specifying an icon. Pinned tags give you a quick way to see your most important tags at a glance; they're displayed at the top of the page when viewing a person's profile (one of: tick, cross, archive, warning, flag, magic, star, profile, user, couple, group, pledge, education, job, gift, role, duration, smiley-happy, smiley-neutral, smiley-sad)"""
    body = _body({"name": name, "category_id": category_id, "description": description, "colour": colour, "is_smart": is_smart, "visible_to": visible_to, "pinned_icon": pinned_icon}, None, (), keep=())
    return await _request("POST", "/addressbook/tags", None, body)

@tool("addressbook", "POST")
async def addressbook_create_tag_resource(
    person: Annotated[dict[str, Any], Field(description="Person (person) - REQUIRED. The Person to link to the Tag. Object keys: {type (one of addressbook_contact, required): The Person's type, indicating whether they are a Child, Contact, Customer or Giver -> For a contact use 'addressbook_contact'.; id (integer, required): The Person's ID -> Contact ID: run addressbook_list_contacts (search with `q`) and use the `id`.}")],
    tag_id: Annotated[int, Field(description="Tag ID (tag_id) - REQUIRED. The Tag's ID. HOW TO GET IT: Run addressbook_list_tags and use the tag `id`.")],
) -> Any:
    """Create a tag resource. Create a tag resource linking a contact to a tag
    
    WHAT YOU NEED TO PROVIDE
      - person (required): The Person to link to the Tag
      - tag_id (required): The Tag's ID -> Run addressbook_list_tags and use the tag `id`."""
    body = _body({"person": person, "tag_id": tag_id}, None, (), keep=())
    return await _request("POST", "/addressbook/tag_resources", None, body)

@tool("addressbook", "PUT")
async def addressbook_edit_contact(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the contact to change. Find it with addressbook_list_contacts (use `q` to search) or the matching get tool.")],
    first_name: Annotated[str | None, Field(description="First name (first_name) - Optional - leave out to keep the current value. The first name of the Contact. (max 50 characters; cannot be empty)")] = None,
    last_name: Annotated[str | None, Field(description="Last name (last_name) - Optional - leave out to keep the current value. The last name of the Contact. (max 50 characters; cannot be empty)")] = None,
    middle_name: Annotated[str | None, Field(description="Middle name (middle_name) - Optional - leave out to keep the current value. The middle name of the Contact. (max 50 characters)")] = None,
    title: Annotated[str | None, Field(description="Title (title) - Optional - leave out to keep the current value. The title of the Contact. (max 25 characters)")] = None,
    formal_name: Annotated[str | None, Field(description="Formal name (formal_name) - Optional - leave out to keep the current value. The formal name of the Contact. (max 50 characters)")] = None,
    former_name: Annotated[str | None, Field(description="Former name (former_name) - Optional - leave out to keep the current value. Use this if a Contact has changed their last name and you want to record their previous last name. I.e. someone's maiden name. (max 50 characters)")] = None,
    email: Annotated[str | None, Field(description="Email (email) - Optional - leave out to keep the current value. The Contact's email address. (max 100 characters; an email address)")] = None,
    mobile: Annotated[str | None, Field(description="Mobile (mobile) - Optional - leave out to keep the current value. The Contact's mobile number. (max 20 characters)")] = None,
    telephone: Annotated[str | None, Field(description="Telephone (telephone) - Optional - leave out to keep the current value. The Contact's telephone number. (max 20 characters)")] = None,
    address: Annotated[dict[str, Any] | None, Field(description="Address (address) - Optional - leave out to keep the current value. Edit a street Address. Object keys: {line_1 (string, required): The first line of the Address; line_2 (string, required): The second line of the Address; line_3 (string, required): The third line of the Address; city (string, required): The city of the Address; county (string, required): The county of the Address; country (string, required): The country of the Address, in ISO3166 alpha 2 format; postcode (string, required): The postcode of the Address}")] = None,
    sex: Annotated[Literal["female", "male"] | None, Field(description="Sex (sex) - Optional - leave out to keep the current value. The sex of the Contact. (one of: female, male)")] = None,
    date_of_birth: Annotated[str | None, Field(description="Date of birth (date_of_birth) - Optional - leave out to keep the current value. The Contact's date of birth. (date as YYYY-MM-DD)")] = None,
    marital_status: Annotated[Literal["single", "married", "engaged", "partnered", "separated", "divorced", "widowed"] | None, Field(description="Marital status (marital_status) - Optional - leave out to keep the current value. The Contact's marital status. (one of: single, married, engaged, partnered, separated, divorced, widowed)")] = None,
    spouse_id: Annotated[int | None, Field(description="Spouse ID (spouse_id) - Optional - leave out to keep the current value. The Contact ID for the Contact's spouse. HOW TO GET IT: Run addressbook_list_contacts and use the spouse's contact `id`.")] = None,
    job: Annotated[str | None, Field(description="Job (job) - Optional - leave out to keep the current value. The job of the Contact. (max 100 characters)")] = None,
    employer: Annotated[str | None, Field(description="Employer (employer) - Optional - leave out to keep the current value. The employer of the Contact. (max 100 characters)")] = None,
    work_telephone: Annotated[str | None, Field(description="Work telephone (work_telephone) - Optional - leave out to keep the current value. The Contact's work telephone number. (max 20 characters)")] = None,
    custom_fields: Annotated[list[dict[str, Any]] | None, Field(description="Custom fields (custom_fields) - Optional - leave out to keep the current value. Custom Field responses for this resource. HOW TO GET IT: Use addressbook_list_custom_fields to find each field's `id` and type. Object keys: list of ONE of these shapes: {id (integer, required): The ID of the Custom Field; field_type (one of text/textarea/phone/date, required): A discriminator to determine the type of Custom Field this response belongs to; value (string, required): The raw value of a text, textarea, phone or date} OR {id (integer, required): The ID of the Custom Field; field_type (one of email, required): A discriminator to determine the type of Custom Field this response belongs to; value (string, required): The raw value of an email} OR {id (integer, required): The ID of the Custom Field; field_type (one of radio/select, required): A discriminator to determine the type of Custom Field this response belongs to; value ({id (string, required)}, required): The UUID of the selected response for a radio or select field} OR {id (integer, required): The ID of the Custom Field; field_type (one of checkbox, required): A discriminator to determine the type of Custom Field this response belongs to; value (list of {id (string, required)}, required): A list of UUIDs of the selected responses for a checkbox field}")] = None,
    communication: Annotated[dict[str, Any] | None, Field(description="Communication (communication) - Optional - leave out to keep the current value. The person's communication preferences. Object keys: {general_email (boolean, required): Whether this person has consented to receiving general emails; general_sms (boolean, required): Whether this person has consented to receiving general SMS messages; phone (boolean, required): Whether this person has consented to receiving phone calls; post (boolean, required): Whether this person has consented to receiving post; rota_email (boolean, required): Whether this person has consented to receiving rota emails; rota_sms (boolean, required): Whether this person has consented to receiving rota SMS messages}")] = None,
    login: Annotated[dict[str, Any] | None, Field(description="Login (login) - Optional - leave out to keep the current value. My ChurchSuite/CharitySuite login information. Object keys: {enabled (boolean, required): Whether the person is able to log into My; invited (boolean, required): Whether the person has been invited to create a My account}")] = None,
    privacy: Annotated[dict[str, Any] | None, Field(description="Privacy (privacy) - Optional - leave out to keep the current value. Edit the Contact's privacy settings. Object keys: {name_visible (boolean, required): Whether the Contact's name is visible in My; address_visible (boolean, required): Whether the Contact's address is visible in My. This will only be respected if name_visible is true, else it will not be visible; mobile_visible (boolean, required): Whether the Contact's mobile is visible in My. This will only be respected if name_visible is true, else it will not be visible; email_visible (boolean, required): Whether the Contact's email is visible in My. This will only be respected if name_visible is true, else it will not be visible; telephone_visible (boolean, required): Whether the Contact's mobile is visible in My. This will only be respected if name_visible is true, else it will not be visible}")] = None,
    all_sites: Annotated[bool | None, Field(description="All sites (all_sites) - Optional - leave out to keep the current value. Whether the Contact belongs to All Sites.")] = None,
    site_ids: Annotated[list[int] | None, Field(description="Site IDs (site_ids) - Optional - leave out to keep the current value. An array of Site IDs that the Contact belongs to. If `all_sites` is `true`, this will be overridden. You are only able to add site IDs for sites you have the relevant permission for. HOW TO GET IT: No list tool for sites. Copy `site_ids` from an existing record (e.g. addressbook_get_contact), or use all_sites=true for every site.")] = None,
    set_null: Annotated[list[Literal["address", "sex", "date_of_birth", "marital_status", "spouse_id"]] | None, Field(description="Optional. Names of fields to CLEAR (set to null). Allowed: address, sex, date_of_birth, marital_status, spouse_id. Leave a field out of both its own parameter and this list to keep it unchanged.")] = None,
) -> Any:
    """Update a contact. Update a single contact identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the contact to change. Find it with addressbook_list_contacts (use `q` to search) or the matching get tool.
      - first_name (optional): The first name of the Contact (max 50 characters; cannot be empty)
      - last_name (optional): The last name of the Contact (max 50 characters; cannot be empty)
      - middle_name (optional): The middle name of the Contact (max 50 characters)
      - title (optional): The title of the Contact (max 25 characters)
      - formal_name (optional): The formal name of the Contact (max 50 characters)
      - former_name (optional): Use this if a Contact has changed their last name and you want to record their previous last name. I.e. someone's maiden name (max 50 characters)
      - email (optional): The Contact's email address (max 100 characters; an email address)
      - mobile (optional): The Contact's mobile number (max 20 characters)
      - telephone (optional): The Contact's telephone number (max 20 characters)
      - address (optional): Edit a street Address
      - sex (optional): The sex of the Contact (one of: female, male)
      - date_of_birth (optional): The Contact's date of birth (date as YYYY-MM-DD)
      - marital_status (optional): The Contact's marital status (one of: single, married, engaged, partnered, separated, divorced, widowed)
      - spouse_id (optional): The Contact ID for the Contact's spouse -> Run addressbook_list_contacts and use the spouse's contact `id`.
      - job (optional): The job of the Contact (max 100 characters)
      - employer (optional): The employer of the Contact (max 100 characters)
      - work_telephone (optional): The Contact's work telephone number (max 20 characters)
      - custom_fields (optional): Custom Field responses for this resource -> Use addressbook_list_custom_fields to find each field's `id` and type.
      - communication (optional): The person's communication preferences
      - login (optional): My ChurchSuite/CharitySuite login information
      - privacy (optional): Edit the Contact's privacy settings
      - all_sites (optional): Whether the Contact belongs to All Sites
      - site_ids (optional): An array of Site IDs that the Contact belongs to. If `all_sites` is `true`, this will be overridden. You are only able to add site IDs for sites you have the relevant permission for -> No list tool for sites. Copy `site_ids` from an existing record (e.g. addressbook_get_contact), or use all_sites=true for every site.
      - set_null (optional): list of fields to clear: address, sex, date_of_birth, marital_status, spouse_id."""
    body = _body({"first_name": first_name, "last_name": last_name, "middle_name": middle_name, "title": title, "formal_name": formal_name, "former_name": former_name, "email": email, "mobile": mobile, "telephone": telephone, "address": address, "sex": sex, "date_of_birth": date_of_birth, "marital_status": marital_status, "spouse_id": spouse_id, "job": job, "employer": employer, "work_telephone": work_telephone, "custom_fields": custom_fields, "communication": communication, "login": login, "privacy": privacy, "all_sites": all_sites, "site_ids": site_ids}, set_null, ("address", "sex", "date_of_birth", "marital_status", "spouse_id", ), keep=())
    return await _request("PUT", f"/addressbook/contacts/{id}", None, body)

@tool("addressbook", "PUT")
async def addressbook_edit_flow_stage(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the flow stage to change. Find it with addressbook_list_flow_stages (use `q` to search) or the matching get tool.")],
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Stage's name. (max 50 characters; cannot be empty)")],
    instructions: Annotated[str, Field(description="Instructions (instructions) - REQUIRED. The instructions for the Stage. (max 65535 characters)")],
    due_interval: Annotated[int, Field(description="Due interval (due_interval) - REQUIRED. The number of days, weeks or months used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage.")],
    due_period: Annotated[Literal["days", "weeks", "months"], Field(description="Due period (due_period) - REQUIRED. The due period used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage. (one of: days, weeks, months)")],
    due_from: Annotated[Literal["first_added", "previous_stage"], Field(description="Due from (due_from) - REQUIRED. The relative point used for tracking a person linked to the Stage, either when they are first added to the flow or when they are processed from a previous stage. (one of: first_added, previous_stage)")],
    assigned_user_id: Annotated[int | None, Field(description="Assigned user ID (assigned_user_id) - REQUIRED. The ID of the user automatically assigned to people in this stage, or null if no user is assigned. HOW TO GET IT: A ChurchSuite user ID. No list tool; it appears as `assigned_user_id` on existing flow stages/trackings.")],
    automate_time: Annotated[str | None, Field(description="Automate time (automate_time) - REQUIRED. The UTC time the Stage will automatically process any due or overdue people if it is automated, or null if it is not automated.")],
) -> Any:
    """Update a flow stage. Update a single stage identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the flow stage to change. Find it with addressbook_list_flow_stages (use `q` to search) or the matching get tool.
      - name (required): The Stage's name (max 50 characters; cannot be empty)
      - instructions (required): The instructions for the Stage (max 65535 characters)
      - due_interval (required): The number of days, weeks or months used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage
      - due_period (required): The due period used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage (one of: days, weeks, months)
      - due_from (required): The relative point used for tracking a person linked to the Stage, either when they are first added to the flow or when they are processed from a previous stage (one of: first_added, previous_stage)
      - assigned_user_id (required): The ID of the user automatically assigned to people in this stage, or null if no user is assigned -> A ChurchSuite user ID. No list tool; it appears as `assigned_user_id` on existing flow stages/trackings.
      - automate_time (required): The UTC time the Stage will automatically process any due or overdue people if it is automated, or null if it is not automated"""
    body = _body({"name": name, "instructions": instructions, "due_interval": due_interval, "due_period": due_period, "due_from": due_from, "assigned_user_id": assigned_user_id, "automate_time": automate_time}, None, (), keep=("assigned_user_id", "automate_time", ))
    return await _request("PUT", f"/addressbook/flow_stages/{id}", None, body)

@tool("addressbook", "PUT")
async def addressbook_edit_flow_tracking(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the flow tracking to change. Find it with addressbook_list_flow_trackings (use `q` to search) or the matching get tool.")],
    due_date: Annotated[str, Field(description="Due date (due_date) - REQUIRED. The Tracking's due date. (date as YYYY-MM-DD)")],
    assigned_user_id: Annotated[int | None, Field(description="Assigned user ID (assigned_user_id) - REQUIRED. The Tracking's assigned user ID, or null if not assigned to a user. HOW TO GET IT: A ChurchSuite user ID. No list tool; it appears as `assigned_user_id` on existing flow stages/trackings.")],
) -> Any:
    """Update a flow tracking. Update a single flow tracking, linked to a person's progress through a flow, identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the flow tracking to change. Find it with addressbook_list_flow_trackings (use `q` to search) or the matching get tool.
      - due_date (required): The Tracking's due date (date as YYYY-MM-DD)
      - assigned_user_id (required): The Tracking's assigned user ID, or null if not assigned to a user -> A ChurchSuite user ID. No list tool; it appears as `assigned_user_id` on existing flow stages/trackings."""
    body = _body({"due_date": due_date, "assigned_user_id": assigned_user_id}, None, (), keep=("assigned_user_id", ))
    return await _request("PUT", f"/addressbook/flow_trackings/{id}", None, body)

@tool("addressbook", "PUT")
async def addressbook_edit_key_date(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the key date to change. Find it with addressbook_list_key_dates (use `q` to search) or the matching get tool.")],
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Key Date's name. (max 50 characters; cannot be empty)")],
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], Field(description="Colour (colour) - REQUIRED. The Key Date's colour. (one of: blue, red, orange, green, purple, pink, brown, grey)")],
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - REQUIRED. Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is false, `user_group_ids` must contain at least one User Group ID. Object keys: ONE of these shapes: {just_me (one of false): The resource should only be visible to the current user; user_group_ids (list of integer, required): List of user group IDs the resource is visible to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.} OR {just_me (one of true, required): The resource should only be visible to the current user; user_group_ids (list of integer): List of user group IDs the resource is visible to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")],
) -> Any:
    """Update a key date. Update a single key date identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the key date to change. Find it with addressbook_list_key_dates (use `q` to search) or the matching get tool.
      - name (required): The Key Date's name (max 50 characters; cannot be empty)
      - colour (required): The Key Date's colour (one of: blue, red, orange, green, purple, pink, brown, grey)
      - visible_to (required): Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is false, `user_group_ids` must contain at least one User Group ID"""
    body = _body({"name": name, "colour": colour, "visible_to": visible_to}, None, (), keep=("visible_to", ))
    return await _request("PUT", f"/addressbook/key_dates/{id}", None, body)

@tool("addressbook", "PUT")
async def addressbook_edit_key_date_resource(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the key date resource to change. Find it with addressbook_list_key_date_resources (use `q` to search) or the matching get tool.")],
    date: Annotated[str, Field(description="Date (date) - REQUIRED. The date of the Key Date Resource. (date as YYYY-MM-DD)")],
    description: Annotated[str, Field(description="Description (description) - REQUIRED. The description of this Key Date Resource. (max 65535 characters)")],
) -> Any:
    """Update a key date resource. Update a single key date resource identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the key date resource to change. Find it with addressbook_list_key_date_resources (use `q` to search) or the matching get tool.
      - date (required): The date of the Key Date Resource (date as YYYY-MM-DD)
      - description (required): The description of this Key Date Resource (max 65535 characters)"""
    body = _body({"date": date, "description": description}, None, (), keep=())
    return await _request("PUT", f"/addressbook/key_date_resources/{id}", None, body)

@tool("addressbook", "PUT")
async def addressbook_edit_tag(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the tag to change. Find it with addressbook_list_tags (use `q` to search) or the matching get tool.")],
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Tag's name. (max 50 characters; cannot be empty)")],
    category_id: Annotated[int | None, Field(description="Category ID (category_id) - REQUIRED. The Tag's Category ID. HOW TO GET IT: Run addressbook_list_tag_categories and use the `id` of the category.")],
    description: Annotated[str, Field(description="Description (description) - REQUIRED. The description of the Tag. (max 65535 characters)")],
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], Field(description="Colour (colour) - REQUIRED. The name of the colour of the Tag. (one of: blue, red, orange, green, purple, pink, brown, grey)")],
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - REQUIRED. Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is false, `user_group_ids` must contain at least one User Group ID. Object keys: ONE of these shapes: {just_me (one of false): The resource should only be visible to the current user; user_group_ids (list of integer, required): List of user group IDs the resource is visible to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.} OR {just_me (one of true, required): The resource should only be visible to the current user; user_group_ids (list of integer): List of user group IDs the resource is visible to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")],
    pinned_icon: Annotated[Literal["tick", "cross", "archive", "warning", "flag", "magic", "star", "profile", "user", "couple", "group", "pledge", "education", "job", "gift", "role", "duration", "smiley-happy", "smiley-neutral", "smiley-sad"] | None, Field(description="Pinned icon (pinned_icon) - REQUIRED. Pin this Tag by specifying an icon. This will be displayed within the application. (one of: tick, cross, archive, warning, flag, magic, star, profile, user, couple, group, pledge, education, job, gift, role, duration, smiley-happy, smiley-neutral, smiley-sad)")],
) -> Any:
    """Update a tag.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the tag to change. Find it with addressbook_list_tags (use `q` to search) or the matching get tool.
      - name (required): The Tag's name (max 50 characters; cannot be empty)
      - category_id (required): The Tag's Category ID -> Run addressbook_list_tag_categories and use the `id` of the category.
      - description (required): The description of the Tag (max 65535 characters)
      - colour (required): The name of the colour of the Tag (one of: blue, red, orange, green, purple, pink, brown, grey)
      - visible_to (required): Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is false, `user_group_ids` must contain at least one User Group ID
      - pinned_icon (required): Pin this Tag by specifying an icon. This will be displayed within the application (one of: tick, cross, archive, warning, flag, magic, star, profile, user, couple, group, pledge, education, job, gift, role, duration, smiley-happy, smiley-neutral, smiley-sad)"""
    body = _body({"name": name, "category_id": category_id, "description": description, "colour": colour, "visible_to": visible_to, "pinned_icon": pinned_icon}, None, (), keep=("category_id", "visible_to", "pinned_icon", ))
    return await _request("PUT", f"/addressbook/tags/{id}", None, body)

@tool("addressbook", "DELETE")
async def addressbook_delete_contact(id: int) -> Any:
    """DESTRUCTIVE. Delete a contact. Delete a single contact identified by ID."""
    return await _request("DELETE", f"/addressbook/contacts/{id}", None)

@tool("addressbook", "DELETE")
async def addressbook_delete_flow_stage(id: int) -> Any:
    """DESTRUCTIVE. Delete a flow stage. Delete a single stage identified by ID. When a stage is deleted, all actions and flow trackings linked to the stage will be deleted."""
    return await _request("DELETE", f"/addressbook/flow_stages/{id}", None)

@tool("addressbook", "DELETE")
async def addressbook_delete_flow_tracking(id: int) -> Any:
    """DESTRUCTIVE. Delete a flow tracking. Delete a single flow tracking, linked to a person's process through a flow, identified by ID."""
    return await _request("DELETE", f"/addressbook/flow_trackings/{id}", None)

@tool("addressbook", "DELETE")
async def addressbook_delete_key_date(id: int) -> Any:
    """DESTRUCTIVE. Delete a key date. Delete a single key date identified by ID."""
    return await _request("DELETE", f"/addressbook/key_dates/{id}", None)

@tool("addressbook", "DELETE")
async def addressbook_delete_key_date_resource(id: int) -> Any:
    """DESTRUCTIVE. Delete a key date resource. Delete a single key date resource identified by an ID, removing a contact from a key date."""
    return await _request("DELETE", f"/addressbook/key_date_resources/{id}", None)

@tool("addressbook", "DELETE")
async def addressbook_delete_tag(id: int) -> Any:
    """DESTRUCTIVE. Delete a tag. Delete a single tag identified by an ID."""
    return await _request("DELETE", f"/addressbook/tags/{id}", None)

@tool("addressbook", "DELETE")
async def addressbook_delete_tag_resource(contact_id: int, tag_id: int) -> Any:
    """DESTRUCTIVE. Delete a tag resource. Delete a single tag resource identified by a contact ID and tag ID, unlinking a contact from a tag.
    contact_id (required): Filter records by the provided ID.
    tag_id (required): Filter records by the provided ID."""
    return await _request("DELETE", "/addressbook/tag_resources", {"contact_id": contact_id, "tag_id": tag_id})



# ======== calendar ========

@tool("calendar", "GET")
async def calendar_get_category(id: int) -> Any:
    """Get a category. Retrieve a single category identified by ID."""
    return await _request("GET", f"/calendar/categories/{id}", None)

@tool("calendar", "GET")
async def calendar_get_event(id: int) -> Any:
    """Get an event. Retrieve a single event identified by ID."""
    return await _request("GET", f"/calendar/events/{id}", None)

@tool("calendar", "GET")
async def calendar_get_event_note(id: int) -> Any:
    """List event notes. Returns a list of event notes, sorted by pinned, created_at."""
    return await _request("GET", f"/calendar/events/{id}/notes", None)

@tool("calendar", "GET")
async def calendar_get_invite(id: int) -> Any:
    """Get an invite. Retrieve a single invite identified by ID."""
    return await _request("GET", f"/calendar/invites/{id}", None)

@tool("calendar", "GET")
async def calendar_get_label(id: int) -> Any:
    """Get a label. Retrieve a single label identified by ID."""
    return await _request("GET", f"/calendar/labels/{id}", None)

@tool("calendar", "GET")
async def calendar_list_categories(page: int | None = None, per_page: int | None = None, status: Literal["active", "archived"] | None = None, q: str | None = None) -> Any:
    """List categories. Returns a list of categories, sorted alphabetically by name.
    status: Filter records by the provided status.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/calendar/categories", {"page": page, "per_page": per_page, "status": status, "q": q})

@tool("calendar", "GET")
async def calendar_list_events(page: int | None = None, per_page: int | None = None, site_ids: list[int] | None = None, category_ids: list[int] | None = None, q: str | None = None, sequence_id: int | None = None, starts_after: str | None = None, starts_before: str | None = None, status: Literal["cancelled", "confirmed", "pending"] | None = None, order_by: Literal["created_at:asc", "created_at:desc", "date:asc", "date:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List events. Returns a list of events, sorted by starts_at date.
    site_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    category_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    q: Filter records with a fuzzy string match against the following properties: name, starts_at.
    sequence_id: Filter records by the provided value.
    starts_after: Only include records that start after this date.
    starts_before: Only include records that start before this date.
    status: Filter records by the provided status.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/calendar/events", {"page": page, "per_page": per_page, "site_ids[]": site_ids, "category_ids[]": category_ids, "q": q, "sequence_id": sequence_id, "starts_after": starts_after, "starts_before": starts_before, "status": status, "order_by": order_by})

@tool("calendar", "GET")
async def calendar_list_invites(child_ids: list[int] | None = None, contact_ids: list[int] | None = None, event_ids: list[int] | None = None, statuses: list[Literal["accepted", "declined", "not_sent", "opened", "pending"]] | None = None, q: str | None = None, page: int | None = None, per_page: int | None = None) -> Any:
    """List invites. Returns a list of invites for an event, sorted by modified_at.
    child_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    contact_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    event_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    statuses: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    q: Filter records with a fuzzy string match against the following properties: contact_first_name, contact_last_name, child_first_name, child_last_name."""
    return await _request("GET", "/calendar/invites", {"child_ids[]": child_ids, "contact_ids[]": contact_ids, "event_ids[]": event_ids, "statuses[]": statuses, "q": q, "page": page, "per_page": per_page})

@tool("calendar", "GET")
async def calendar_list_label_resources(page: int | None = None, per_page: int | None = None, label_ids: list[int] | None = None, event_ids: list[int] | None = None) -> Any:
    """List label resources. Returns a list of label resources
    label_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    event_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/calendar/label_resources", {"page": page, "per_page": per_page, "label_ids[]": label_ids, "event_ids[]": event_ids})

@tool("calendar", "GET")
async def calendar_list_labels(page: int | None = None, per_page: int | None = None, q: str | None = None) -> Any:
    """List labels. Returns a list of labels, sorted by their order property.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/calendar/labels", {"page": page, "per_page": per_page, "q": q})

@tool("calendar", "GET")
async def calendar_list_signups(event_id: int | None = None, status: Literal["active", "cancelled", "refunded", "reserved"] | None = None, page: int | None = None, per_page: int | None = None) -> Any:
    """List sign-ups. Returns a list of sign-ups, sorted by ID.
    event_id: Filter records by the provided ID.
    status: Filter records by the provided status."""
    return await _request("GET", "/calendar/signups", {"event_id": event_id, "status": status, "page": page, "per_page": per_page})

@tool("calendar", "GET")
async def calendar_list_tickets(event_id: int | None = None, page: int | None = None, per_page: int | None = None) -> Any:
    """List tickets. Returns a list of tickets, sorted by ID.
    event_id: Filter records by the provided ID."""
    return await _request("GET", "/calendar/tickets", {"event_id": event_id, "page": page, "per_page": per_page})

@tool("calendar", "POST")
async def calendar_create_category(
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Category's name. (max 50 characters; cannot be empty)")],
    color: Annotated[str, Field(description="Color (color) - REQUIRED. The Category's hexadecimal colour value.")],
    brand_id: Annotated[int, Field(description="Brand ID (brand_id) - REQUIRED. The Brand ID linked to the Category. HOW TO GET IT: No list tool. Copy `brand_id` from an existing category or event (calendar_get_category / calendar_get_event).")],
) -> Any:
    """Create a category. Create a category with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - name (required): The Category's name (max 50 characters; cannot be empty)
      - color (required): The Category's hexadecimal colour value
      - brand_id (required): The Brand ID linked to the Category -> No list tool. Copy `brand_id` from an existing category or event (calendar_get_category / calendar_get_event)."""
    body = _body({"name": name, "color": color, "brand_id": brand_id}, None, (), keep=())
    return await _request("POST", "/calendar/categories", None, body)

@tool("calendar", "POST")
async def calendar_create_event(
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Event's name. (max 50 characters; cannot be empty)")],
    starts_at: Annotated[str, Field(description="Starts at (starts_at) - REQUIRED. Date and time the Event starts. (ISO 8601 date-time. Give NZ local time with no Z or offset, e.g. 2026-11-01T19:30:00 for 7:30pm; a value with Z or an offset is converted to the account's local time first)")],
    ends_at: Annotated[str, Field(description="Ends at (ends_at) - REQUIRED. Date and time the Event ends. (ISO 8601 date-time. Give NZ local time with no Z or offset, e.g. 2026-11-01T19:30:00 for 7:30pm; a value with Z or an offset is converted to the account's local time first)")],
    status: Annotated[Literal["confirmed", "cancelled", "pending"], Field(description="Status (status) - REQUIRED. The current status of the Event. (one of: confirmed, cancelled, pending)")],
    category_id: Annotated[int, Field(description="Category ID (category_id) - REQUIRED. The Event's category ID. HOW TO GET IT: Run calendar_list_categories and use the `id` of the category (match on `name`).")],
    sequence_id: Annotated[int | None, Field(description="Sequence ID (sequence_id) - Optional. If the Event is part of a sequence, this field should contain the sequence ID from an existing Event in the sequence. HOW TO GET IT: Only for repeating events: copy `sequence_id` from another event in the same series (calendar_get_event or calendar_list_events). Leave out for a one-off event.")] = None,
    description: Annotated[str | None, Field(description="Description (description) - Optional. The description of the Event. (cannot be empty)")] = None,
    all_sites: Annotated[bool | None, Field(description="All sites (all_sites) - Optional. Whether the Event belongs to All Sites. Only true is accepted. Note: on Trinity's account this is saved as site_ids [1] (the only site), not all_sites=true. Pass this OR site_ids - never both.")] = None,
    site_ids: Annotated[list[int] | None, Field(description="Site IDs (site_ids) - Optional. List of all site IDs the Event belongs to. HOW TO GET IT: No list tool for sites. Copy `site_ids` from an existing record (e.g. calendar_get_event), or use all_sites=true for every site. Pass this OR all_sites - never both.")] = None,
) -> Any:
    """Create an event. Create an event with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - name (required): The Event's name (max 50 characters; cannot be empty)
      - starts_at (required): Date and time the Event starts (ISO 8601 date-time. Give NZ local time with no Z or offset, e.g. 2026-11-01T19:30:00 for 7:30pm; a value with Z or an offset is converted to the account's local time first)
      - ends_at (required): Date and time the Event ends (ISO 8601 date-time. Give NZ local time with no Z or offset, e.g. 2026-11-01T19:30:00 for 7:30pm; a value with Z or an offset is converted to the account's local time first)
      - status (required): The current status of the Event (one of: confirmed, cancelled, pending)
      - category_id (required): The Event's category ID -> Run calendar_list_categories and use the `id` of the category (match on `name`).
      - sequence_id (optional): If the Event is part of a sequence, this field should contain the sequence ID from an existing Event in the sequence -> Only for repeating events: copy `sequence_id` from another event in the same series (calendar_get_event or calendar_list_events). Leave out for a one-off event.
      - description (optional): The description of the Event (cannot be empty)
      - all_sites (optional): Whether the Event belongs to All Sites
      - site_ids (optional): List of all site IDs the Event belongs to -> No list tool for sites. Copy `site_ids` from an existing record (e.g. calendar_get_event), or use all_sites=true for every site.
      Site rule: pass exactly one of: all_sites OR site_ids.
    
    TIMES: ChurchSuite reads starts_at/ends_at as the account's local (NZ) time. Pass local time with no Z, e.g. 2026-11-01T19:30:00. Values come back from ChurchSuite in true UTC, so 19:30 NZ shows as 06:30Z. Always check the returned event.

    NOT SETTABLE VIA THE API (v2.112.3), set these inside ChurchSuite afterwards:
      - visibility (My ChurchSuite, website/Embed, Connect, featured): read-only. New events get the Calendar module's default, which on Trinity's account is hidden everywhere.
      - location, image, sign-up options and confirmation email.
    REPEATING EVENTS: this cannot start a new series. sequence_id only adds an event to a series that already exists; otherwise create one event per date."""
    starts_at, ends_at = _to_account_local(starts_at), _to_account_local(ends_at)
    body = _body({"sequence_id": sequence_id, "name": name, "starts_at": starts_at, "ends_at": ends_at, "status": status, "description": description, "category_id": category_id, "all_sites": all_sites, "site_ids": site_ids}, None, (), keep=())
    if all_sites is not None and all_sites is not True:
        raise ValueError("all_sites can only be true. To limit the item to certain sites pass site_ids instead.")
    _check_choice(body, [(['all_sites'], ['all_sites']), (['site_ids'], ['site_ids'])], True)
    return await _request("POST", "/calendar/events", None, body)

@tool("calendar", "POST")
async def calendar_create_event_note(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the event to add the note to. Find it with calendar_list_events (use `q` to search).")],
    content: Annotated[str, Field(description="Content (content) - REQUIRED. The content of the Note. (max 65535 characters; cannot be empty)")],
    pinned: Annotated[bool | None, Field(description="Pinned (pinned) - Optional. Whether the Note should be pinned.")] = None,
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - Optional. Object keys: ONE of these shapes: {just_me (one of true): The resource should only be visible to the current user} OR {user_group_ids (list of integer): List of user group IDs the resource is visible to. You are only able to add user group IDs for user groups you have access to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")] = None,
) -> Any:
    """Create an event note. Create a Note against an Event in the Calendar module.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the event to add the note to. Find it with calendar_list_events (use `q` to search).
      - content (required): The content of the Note (max 65535 characters; cannot be empty)
      - pinned (optional): Whether the Note should be pinned
      - visible_to (optional): Visible to"""
    body = _body({"content": content, "pinned": pinned, "visible_to": visible_to}, None, (), keep=())
    return await _request("POST", f"/calendar/events/{id}/notes", None, body)

@tool("calendar", "POST")
async def calendar_create_signup(event_id: int, person: dict[str, Any] | None = None, first_name: str | None = None, last_name: str | None = None, ticket_id: int | None = None, email: str | None = None, mobile: str | None = None, notes: str | None = None, paid_amount: int | None = None, question_responses: list[dict[str, Any]] | None = None, send_confirmation: bool | None = None, override_capacity: bool | None = None) -> Any:
    """Create a sign-up. Create a sign-up with the provided settings. Two forms: (a) linked to a ChurchSuite person: pass person; (b) not linked: pass first_name and last_name (do not pass person).
    event_id (required): The Event ID the Sign-Up is linked to.
    person: Link to a ChurchSuite person. Keys: type ("addressbook_contact" or "children_child"), id. Required unless first_name and last_name are given.
    first_name: The Sign-Up's first name (1-50 chars). Required with last_name when person is not given.
    last_name: The Sign-Up's last name (1-50 chars). Required with first_name when person is not given.
    ticket_id: The Ticket ID the Sign-Up is linked to. Required if the Event has tickets.
    email: The Sign-Up's email address (max 100). If omitted and linked to a Contact or Child, theirs is used.
    mobile: The Sign-Up's mobile number (max 20). If omitted and linked to a Contact or Child, theirs is used.
    notes: The Sign-Up's notes (max 65535).
    paid_amount: Amount paid in the smallest currency unit (pence/cents). Required if the Event has tickets.
    question_responses: Items of {question_id: int, question_type, value}. question_type "checkboxes" takes value as a list of strings; "date", "email", "paragraph", "text", "phone", "radios", "select" take a string value.
    send_confirmation: Whether to send a confirmation email to the Sign-Up.
    override_capacity: Whether to override the Event's capacity if this Sign-Up would exceed it."""
    if person is not None and (first_name is not None or last_name is not None):
        raise ValueError("Pass either person, or first_name and last_name, not both")
    if person is None and (first_name is None or last_name is None):
        raise ValueError("Pass either person, or both first_name and last_name")
    return await _request("POST", "/calendar/signups", None, _body({"event_id": event_id, "person": person, "first_name": first_name, "last_name": last_name, "ticket_id": ticket_id, "email": email, "mobile": mobile, "notes": notes, "paid_amount": paid_amount, "question_responses": question_responses, "send_confirmation": send_confirmation, "override_capacity": override_capacity}, None, ()))

@tool("calendar", "PUT")
async def calendar_edit_category(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the category to change. Find it with calendar_list_categories (use `q` to search) or the matching get tool.")],
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Category's name. (max 50 characters; cannot be empty)")],
    color: Annotated[str, Field(description="Color (color) - REQUIRED. The Category's hexadecimal colour value.")],
    brand_id: Annotated[int, Field(description="Brand ID (brand_id) - REQUIRED. The Brand ID linked to the Category. HOW TO GET IT: No list tool. Copy `brand_id` from an existing category or event (calendar_get_category / calendar_get_event).")],
) -> Any:
    """Update a category. Update a single category identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the category to change. Find it with calendar_list_categories (use `q` to search) or the matching get tool.
      - name (required): The Category's name (max 50 characters; cannot be empty)
      - color (required): The Category's hexadecimal colour value
      - brand_id (required): The Brand ID linked to the Category -> No list tool. Copy `brand_id` from an existing category or event (calendar_get_category / calendar_get_event)."""
    body = _body({"name": name, "color": color, "brand_id": brand_id}, None, (), keep=())
    return await _request("PUT", f"/calendar/categories/{id}", None, body)

@tool("calendar", "PUT")
async def calendar_edit_event(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the event to change. Find it with calendar_list_events (use `q` to search) or the matching get tool.")],
    sequence_id: Annotated[int | None, Field(description="Sequence ID (sequence_id) - Optional - leave out to keep the current value. If the Event is part of a sequence, this field should contain the sequence ID from an existing Event in the sequence. HOW TO GET IT: Only for repeating events: copy `sequence_id` from another event in the same series (calendar_get_event or calendar_list_events). Leave out for a one-off event.")] = None,
    name: Annotated[str | None, Field(description="Name (name) - Optional - leave out to keep the current value. The Event's name. (max 50 characters; cannot be empty)")] = None,
    starts_at: Annotated[str | None, Field(description="Starts at (starts_at) - Optional - leave out to keep the current value. Date and time the Event starts. (ISO 8601 date-time. Give NZ local time with no Z or offset, e.g. 2026-11-01T19:30:00 for 7:30pm; a value with Z or an offset is converted to the account's local time first)")] = None,
    ends_at: Annotated[str | None, Field(description="Ends at (ends_at) - Optional - leave out to keep the current value. Date and time the Event ends. (ISO 8601 date-time. Give NZ local time with no Z or offset, e.g. 2026-11-01T19:30:00 for 7:30pm; a value with Z or an offset is converted to the account's local time first)")] = None,
    status: Annotated[Literal["confirmed", "cancelled", "pending"] | None, Field(description="Status (status) - Optional - leave out to keep the current value. The current status of the Event. (one of: confirmed, cancelled, pending)")] = None,
    description: Annotated[str | None, Field(description="Description (description) - Optional - leave out to keep the current value. The description of the Event.")] = None,
    category_id: Annotated[int | None, Field(description="Category ID (category_id) - Optional - leave out to keep the current value. The Event's category ID. HOW TO GET IT: Run calendar_list_categories and use the `id` of the category (match on `name`).")] = None,
    all_sites: Annotated[bool | None, Field(description="All sites (all_sites) - Optional - leave out to keep the current value. Whether the Event belongs to All Sites. Only true is accepted. Note: on Trinity's account this is saved as site_ids [1] (the only site), not all_sites=true. Pass this OR site_ids - never both.")] = None,
    site_ids: Annotated[list[int] | None, Field(description="Site IDs (site_ids) - Optional - leave out to keep the current value. List of all site IDs the Event belongs to. HOW TO GET IT: No list tool for sites. Copy `site_ids` from an existing record (e.g. calendar_get_event), or use all_sites=true for every site. Pass this OR all_sites - never both.")] = None,
    set_null: Annotated[list[Literal["sequence_id"]] | None, Field(description="Optional. Names of fields to CLEAR (set to null). Allowed: sequence_id. Leave a field out of both its own parameter and this list to keep it unchanged.")] = None,
    fill_from_current: Annotated[bool, Field(description="Default true. Read the event first and re-send its unchanged fields, so you only have to pass what you are changing. Set false to send exactly and only the fields you pass.")] = True,
) -> Any:
    """Update an event. Update a single event identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the event to change. Find it with calendar_list_events (use `q` to search) or the matching get tool.
      - sequence_id (optional): If the Event is part of a sequence, this field should contain the sequence ID from an existing Event in the sequence -> Only for repeating events: copy `sequence_id` from another event in the same series (calendar_get_event or calendar_list_events). Leave out for a one-off event.
      - name (optional): The Event's name (max 50 characters; cannot be empty)
      - starts_at (optional): Date and time the Event starts (ISO 8601 date-time. Give NZ local time with no Z or offset, e.g. 2026-11-01T19:30:00 for 7:30pm; a value with Z or an offset is converted to the account's local time first)
      - ends_at (optional): Date and time the Event ends (ISO 8601 date-time. Give NZ local time with no Z or offset, e.g. 2026-11-01T19:30:00 for 7:30pm; a value with Z or an offset is converted to the account's local time first)
      - status (optional): The current status of the Event (one of: confirmed, cancelled, pending)
      - description (optional): The description of the Event
      - category_id (optional): The Event's category ID -> Run calendar_list_categories and use the `id` of the category (match on `name`).
      - all_sites (optional): Whether the Event belongs to All Sites
      - site_ids (optional): List of all site IDs the Event belongs to -> No list tool for sites. Copy `site_ids` from an existing record (e.g. calendar_get_event), or use all_sites=true for every site.
      - set_null (optional): list of fields to clear: sequence_id.
      - fill_from_current (optional): see parameter description.
      Site rule: pass exactly one of: all_sites OR site_ids.
    
    TIMES: pass NZ local time with no Z (e.g. 2026-11-01T19:30:00). When re-sending the current start/end (fill_from_current), the tool converts ChurchSuite's UTC values back to local time so the event does not move.

    VISIBILITY: the API cannot set `visibility` (embed, featured, My ChurchSuite groups); this tool never sends it, so existing visibility is left as it was. Location, image and sign-up options cannot be changed via the API either. Check with calendar_get_event afterwards. By default (fill_from_current=true) the tool first reads the event and re-sends the unchanged name, dates, status, category, description, sequence_id and sites, because ChurchSuite validates a full event on every edit - so you only need to pass what you want to change."""
    starts_at, ends_at = _to_account_local(starts_at), _to_account_local(ends_at)
    body = _body({"sequence_id": sequence_id, "name": name, "starts_at": starts_at, "ends_at": ends_at, "status": status, "description": description, "category_id": category_id, "all_sites": all_sites, "site_ids": site_ids}, set_null, ("sequence_id", ), keep=())
    if all_sites is not None and all_sites is not True:
        raise ValueError("all_sites can only be true. To limit the item to certain sites pass site_ids instead.")
    _check_choice(body, [(['all_sites'], []), (['site_ids'], [])], False)
    if fill_from_current:
        cur = (await _request("GET", f"/calendar/events/{id}")).get("data", {})
        for k in ("name", "starts_at", "ends_at", "status", "category_id", "description"):
            if k not in body and cur.get(k) is not None:
                # ChurchSuite returns true UTC but reads input as local time: convert or the event shifts.
                body[k] = _to_account_local(cur[k]) if k in ("starts_at", "ends_at") else cur[k]
        if "sequence_id" not in body:
            body["sequence_id"] = cur.get("sequence_id")
        if "all_sites" not in body and "site_ids" not in body:
            if cur.get("all_sites"):
                body["all_sites"] = True
            else:
                body["site_ids"] = cur.get("site_ids") or []
    return await _request("PUT", f"/calendar/events/{id}", None, body)

@tool("calendar", "PUT")
async def calendar_edit_signup(id: int, email: str, mobile: str, notes: str, question_responses: list[dict[str, Any]], ticket_id: int | None = None, paid_amount: int | None = None, person: dict[str, Any] | None = None, first_name: str | None = None, last_name: str | None = None) -> Any:
    """Update a sign-up. Update a single sign-up identified by ID. This is a full replacement: every field is sent. Two forms: (a) linked to a ChurchSuite person: pass person (first/last name are sent as null); (b) not linked: pass first_name and last_name (person is sent as null).
    email (required): The Sign-Up's email address (max 100).
    mobile (required): The Sign-Up's mobile number (max 20).
    notes (required): The Sign-Up's notes (max 65535).
    question_responses (required): Items of {question_id: int, question_type, value}; pass [] for none. question_type "checkboxes" takes value as a list of strings; "date", "email", "paragraph", "text", "phone", "radios", "select" take a string value.
    ticket_id: The Ticket ID. Cannot be null if the Event has tickets; null otherwise.
    paid_amount: Amount paid in the smallest currency unit. Cannot be null if the Event has tickets; null otherwise.
    person: Link to a ChurchSuite person. Keys: type ("addressbook_contact" or "children_child"), id.
    first_name: First name (1-50 chars) when the Sign-Up is not linked to a person.
    last_name: Last name (1-50 chars) when the Sign-Up is not linked to a person."""
    if person is not None and (first_name is not None or last_name is not None):
        raise ValueError("Pass either person, or first_name and last_name, not both")
    if person is None and (first_name is None or last_name is None):
        raise ValueError("Pass either person, or both first_name and last_name")
    return await _request("PUT", f"/calendar/signups/{id}", None, _body({"ticket_id": ticket_id, "person": person, "first_name": first_name, "last_name": last_name, "email": email, "mobile": mobile, "notes": notes, "paid_amount": paid_amount, "question_responses": question_responses}, None, (), keep=("ticket_id", "person", "first_name", "last_name", "paid_amount", )))

@tool("calendar", "DELETE")
async def calendar_delete_category(id: int, replacement_id: int) -> Any:
    """DESTRUCTIVE. Delete a category. Delete a single category identified by ID. When a category is deleted, all events linked must be transferred to another category.
    replacement_id (required): The ID of the replacement record. All events linked to the record being deleted will be moved to the replacement record."""
    return await _request("DELETE", f"/calendar/categories/{id}", {"replacement_id": replacement_id})

@tool("calendar", "DELETE")
async def calendar_delete_event(id: int) -> Any:
    """DESTRUCTIVE. Delete an event. Delete a single event identified by ID."""
    return await _request("DELETE", f"/calendar/events/{id}", None)

@tool("calendar", "DELETE")
async def calendar_delete_signup(id: int) -> Any:
    """DESTRUCTIVE. Delete a sign-up. Delete a single sign-up identified by an ID."""
    return await _request("DELETE", f"/calendar/signups/{id}", None)

@tool("calendar", "POST")
async def calendar_create_label(
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Label's name. (max 50 characters; cannot be empty)")],
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], Field(description="Colour (colour) - REQUIRED. The Label's colour. (one of: blue, red, orange, green, purple, pink, brown, grey)")],
    options: Annotated[list[str], Field(description="Options (options) - REQUIRED. List of options that belong to the Label.")],
    multiple_responses: Annotated[bool, Field(description="Multiple responses (multiple_responses) - REQUIRED. Whether the Label can have multiple responses.")],
    required: Annotated[bool, Field(description="Required (required) - REQUIRED. Whether the Label is required.")],
) -> Any:
    """Create a label. Create a label with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - name (required): The Label's name (max 50 characters; cannot be empty)
      - colour (required): The Label's colour (one of: blue, red, orange, green, purple, pink, brown, grey)
      - options (required): List of options that belong to the Label
      - multiple_responses (required): Whether the Label can have multiple responses
      - required (required): Whether the Label is required"""
    body = _body({"name": name, "colour": colour, "options": options, "multiple_responses": multiple_responses, "required": required}, None, (), keep=())
    return await _request("POST", "/calendar/labels", None, body)

@tool("calendar", "PUT")
async def calendar_edit_label(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the label to change. Find it with calendar_list_labels (use `q` to search) or the matching get tool.")],
    name: Annotated[str | None, Field(description="Name (name) - Optional - leave out to keep the current value. The Label's name. (max 50 characters; cannot be empty)")] = None,
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"] | None, Field(description="Colour (colour) - Optional - leave out to keep the current value. The Label's colour. (one of: blue, red, orange, green, purple, pink, brown, grey)")] = None,
    multiple_responses: Annotated[bool | None, Field(description="Multiple responses (multiple_responses) - Optional - leave out to keep the current value. Whether the Label can have multiple responses.")] = None,
    required: Annotated[bool | None, Field(description="Required (required) - Optional - leave out to keep the current value. Whether the Label is required.")] = None,
) -> Any:
    """Update a label. Update a single label identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the label to change. Find it with calendar_list_labels (use `q` to search) or the matching get tool.
      - name (optional): The Label's name (max 50 characters; cannot be empty)
      - colour (optional): The Label's colour (one of: blue, red, orange, green, purple, pink, brown, grey)
      - multiple_responses (optional): Whether the Label can have multiple responses
      - required (optional): Whether the Label is required"""
    body = _body({"name": name, "colour": colour, "multiple_responses": multiple_responses, "required": required}, None, (), keep=())
    return await _request("PUT", f"/calendar/labels/{id}", None, body)

@tool("calendar", "DELETE")
async def calendar_delete_label(id: int) -> Any:
    """DESTRUCTIVE. Delete a label. Delete a single label identified by ID. Label responses will be deleted from all events. Module settings that reference this label will be affected."""
    return await _request("DELETE", f"/calendar/labels/{id}", None)



# ======== children ========

@tool("children", "GET")
async def children_get_child(id: int) -> Any:
    """Get a child. Retrieve a single child identified by ID."""
    return await _request("GET", f"/children/children/{id}", None)

@tool("children", "GET")
async def children_get_child_note(id: int) -> Any:
    """List child notes. Returns a list of child notes, sorted by pinned, created_at."""
    return await _request("GET", f"/children/children/{id}/notes", None)

@tool("children", "GET")
async def children_get_custom_field(id: int) -> Any:
    """Get a custom field. Retrieve a single custom field identified by ID."""
    return await _request("GET", f"/children/custom_fields/{id}", None)

@tool("children", "GET")
async def children_get_flow(id: int) -> Any:
    """Get a flow. Retrieve a single flow identified by ID."""
    return await _request("GET", f"/children/flows/{id}", None)

@tool("children", "GET")
async def children_get_flow_stage(id: int) -> Any:
    """Get a flow stage. Retrieve a single stage identified by ID."""
    return await _request("GET", f"/children/flow_stages/{id}", None)

@tool("children", "GET")
async def children_get_flow_tracking(id: int) -> Any:
    """Get a flow tracking. Retrieve a single flow tracking, linked to a person's process through a flow, identified by ID."""
    return await _request("GET", f"/children/flow_trackings/{id}", None)

@tool("children", "GET")
async def children_get_form_field(id: int) -> Any:
    """Get a form field. Retrieve a single form field identified by ID."""
    return await _request("GET", f"/children/form_fields/{id}", None)

@tool("children", "GET")
async def children_get_form_response(id: int) -> Any:
    """Get a form response. Retrieve a single form response identified by ID."""
    return await _request("GET", f"/children/form_responses/{id}", None)

@tool("children", "GET")
async def children_get_group(id: int) -> Any:
    """Get a group. Retrieve a single group identified by ID."""
    return await _request("GET", f"/children/groups/{id}", None)

@tool("children", "GET")
async def children_get_group_setup(id: int) -> Any:
    """Get a group setup. Retrieve a single group setup identified by ID."""
    return await _request("GET", f"/children/group_setups/{id}", None)

@tool("children", "GET")
async def children_get_key_date(id: int) -> Any:
    """Get a key date. Retrieve a single key date identified by ID."""
    return await _request("GET", f"/children/key_dates/{id}", None)

@tool("children", "GET")
async def children_get_tag(id: int) -> Any:
    """Get a tag. Retrieve a single tag identified by ID."""
    return await _request("GET", f"/children/tags/{id}", None)

@tool("children", "GET")
async def children_get_tag_category(id: int) -> Any:
    """Get a tag category. Retrieve a single tag category identified by ID"""
    return await _request("GET", f"/children/tag_categories/{id}", None)

@tool("children", "GET")
async def children_list_children(page: int | None = None, per_page: int | None = None, site_ids: list[int] | None = None, status: Literal["active", "archived", "pending"] | None = None, group_ids: list[int] | None = None, q: str | None = None, tag_ids: list[int] | None = None, name_visible: bool | None = None, address_visible: bool | None = None, mobile_visible: bool | None = None, email_visible: bool | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List children. Returns a list of children.
    site_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    status: Filter records by the provided status.
    group_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    q: Filter records with a fuzzy string match against the following properties: first_name, last_name, formal_name, email, mobile, address.
    tag_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    name_visible: Filter records to those matching marked as having their name visible in My ChurchSuite.
    address_visible: Filter records to those matching marked as having their address visible in My ChurchSuite.
    mobile_visible: Filter records to those matching marked as having their mobile visible in My ChurchSuite.
    email_visible: Filter records to those matching marked as having their email visible in My ChurchSuite.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/children/children", {"page": page, "per_page": per_page, "site_ids[]": site_ids, "status": status, "group_ids[]": group_ids, "q": q, "tag_ids[]": tag_ids, "name_visible": name_visible, "address_visible": address_visible, "mobile_visible": mobile_visible, "email_visible": email_visible, "order_by": order_by})

@tool("children", "GET")
async def children_list_custom_fields(page: int | None = None, per_page: int | None = None) -> Any:
    """List custom fields. Returns a list of custom fields"""
    return await _request("GET", "/children/custom_fields", {"page": page, "per_page": per_page})

@tool("children", "GET")
async def children_list_flow_stages(page: int | None = None, per_page: int | None = None, flow_ids: list[int] | None = None) -> Any:
    """List flow stages. Returns a list of stages.
    flow_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/children/flow_stages", {"page": page, "per_page": per_page, "flow_ids[]": flow_ids})

@tool("children", "GET")
async def children_list_flow_trackings(page: int | None = None, per_page: int | None = None, due_before: str | None = None, due_after: str | None = None, flow_ids: list[int] | None = None, child_ids: list[int] | None = None, stage_ids: list[int] | None = None) -> Any:
    """List flow trackings. Returns a list of flow trackings linked to people's progress through a flow.
    due_before: Only include records that start before this date.
    due_after: Only include records that start after this date.
    flow_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    child_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    stage_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/children/flow_trackings", {"page": page, "per_page": per_page, "due_before": due_before, "due_after": due_after, "flow_ids[]": flow_ids, "child_ids[]": child_ids, "stage_ids[]": stage_ids})

@tool("children", "GET")
async def children_list_flows(page: int | None = None, per_page: int | None = None, status: Literal["active", "archived"] | None = None, q: str | None = None) -> Any:
    """List flows. Returns a list of flows, sorted alphabetically by name.
    status: Filter Flows by their status.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/children/flows", {"page": page, "per_page": per_page, "status": status, "q": q})

@tool("children", "GET")
async def children_list_form_fields(page: int | None = None, per_page: int | None = None, form_ids: list[int] | None = None) -> Any:
    """List form fields. Returns a list of form fields
    form_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/children/form_fields", {"page": page, "per_page": per_page, "form_ids[]": form_ids})

@tool("children", "GET")
async def children_list_form_responses(page: int | None = None, per_page: int | None = None, form_ids: list[int] | None = None, statuses: list[Literal["active", "archived", "spam"]] | None = None, q: str | None = None, linked_response: bool | None = None, invite_response: bool | None = None) -> Any:
    """List form responses. Returns a list of form responses, sorted alphabetically by name.
    form_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    statuses: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    q: Filter records with a fuzzy string match against the following properties: first_name, last_name.
    linked_response: Filter records by the provided status.
    invite_response: Filter records by the provided status."""
    return await _request("GET", "/children/form_responses", {"page": page, "per_page": per_page, "form_ids[]": form_ids, "statuses[]": statuses, "q": q, "linked_response": linked_response, "invite_response": invite_response})

@tool("children", "GET")
async def children_list_group_members(page: int | None = None, per_page: int | None = None, child_ids: list[int] | None = None, group_setup_ids: list[int] | None = None, group_ids: list[int] | None = None) -> Any:
    """List group members. Returns a list of group members, sorted alphabetically by name.
    child_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    group_setup_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    group_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/children/group_members", {"page": page, "per_page": per_page, "child_ids[]": child_ids, "group_setup_ids[]": group_setup_ids, "group_ids[]": group_ids})

@tool("children", "GET")
async def children_list_group_setups(page: int | None = None, per_page: int | None = None, site_ids: list[int] | None = None, status: Literal["active", "archived"] | None = None) -> Any:
    """List group setups. Returns a list of group setups, sorted alphabetically by name.
    site_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    status: Filter records by the provided status."""
    return await _request("GET", "/children/group_setups", {"page": page, "per_page": per_page, "site_ids[]": site_ids, "status": status})

@tool("children", "GET")
async def children_list_groups(page: int | None = None, per_page: int | None = None, group_setup_ids: list[int] | None = None, status: Literal["active", "archived"] | None = None, q: str | None = None) -> Any:
    """List groups. Returns a list of groups, sorted alphabetically by name.
    group_setup_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    status: Filter records by the provided status.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/children/groups", {"page": page, "per_page": per_page, "group_setup_ids[]": group_setup_ids, "status": status, "q": q})

@tool("children", "GET")
async def children_list_key_date_resources(page: int | None = None, per_page: int | None = None, key_date_ids: list[int] | None = None, child_ids: list[int] | None = None, order_by: Literal["created_at:asc", "created_at:desc", "date:asc", "date:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc"] | None = None) -> Any:
    """List key date resources. Returns a list of key date resources, optionally filtered by key date or child ID. A key date resource records an instance of a key date for a child on a given date.
    key_date_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    child_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/children/key_date_resources", {"page": page, "per_page": per_page, "key_date_ids[]": key_date_ids, "child_ids[]": child_ids, "order_by": order_by})

@tool("children", "GET")
async def children_list_key_dates(page: int | None = None, per_page: int | None = None, q: str | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List key dates. Returns a list of key dates, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/children/key_dates", {"page": page, "per_page": per_page, "q": q, "order_by": order_by})

@tool("children", "GET")
async def children_list_parent_carer_relationships(page: int | None = None, per_page: int | None = None, child_ids: list[int] | None = None) -> Any:
    """List parent/carer relationships. Returns a list of parent/carer relationships.
    child_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/children/parent_carer_relationships", {"page": page, "per_page": per_page, "child_ids[]": child_ids})

@tool("children", "GET")
async def children_list_tag_categories() -> Any:
    """List tag categories. Returns a list of tag categories, sorted alphabetically by name."""
    return await _request("GET", "/children/tag_categories", None)

@tool("children", "GET")
async def children_list_tag_resources(page: int | None = None, per_page: int | None = None, child_ids: list[int] | None = None, tag_ids: list[int] | None = None) -> Any:
    """List tag resources. Returns a list of tag resources. A tag resource is the relational model linking a child to a tag.
    child_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    tag_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/children/tag_resources", {"page": page, "per_page": per_page, "child_ids[]": child_ids, "tag_ids[]": tag_ids})

@tool("children", "GET")
async def children_list_tags(page: int | None = None, per_page: int | None = None, q: str | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List tags. Returns a list of tags, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/children/tags", {"page": page, "per_page": per_page, "q": q, "order_by": order_by})

@tool("children", "POST")
async def children_create_child(
    first_name: Annotated[str, Field(description="First name (first_name) - REQUIRED. The first name of the Child. (max 50 characters; cannot be empty)")],
    last_name: Annotated[str, Field(description="Last name (last_name) - REQUIRED. The last name of the Child. (max 50 characters; cannot be empty)")],
    all_sites: Annotated[bool, Field(description="All sites (all_sites) - REQUIRED. Whether the Child belongs to All Sites.")],
    site_ids: Annotated[list[int], Field(description="Site IDs (site_ids) - REQUIRED. An array of Site IDs that the Child belongs to. If `all_sites` is `true`, this will be overridden. You are only able to add site IDs for sites you have the relevant permission for. HOW TO GET IT: No list tool for sites. Copy `site_ids` from an existing record (e.g. children_get_child), or use all_sites=true for every site.")],
    middle_name: Annotated[str | None, Field(description="Middle name (middle_name) - Optional. The middle name of the Child. (max 50 characters; cannot be empty)")] = None,
    formal_name: Annotated[str | None, Field(description="Formal name (formal_name) - Optional. The formal name of the Child. (max 50 characters; cannot be empty)")] = None,
    sex: Annotated[Literal["female", "male"] | None, Field(description="Sex (sex) - Optional. The sex of the Child. (one of: female, male)")] = None,
    date_of_birth: Annotated[str | None, Field(description="Date of birth (date_of_birth) - Optional. The Child's date of birth. (date as YYYY-MM-DD)")] = None,
    mobile: Annotated[str | None, Field(description="Mobile (mobile) - Optional. The Child's mobile number. (max 20 characters; cannot be empty)")] = None,
    email: Annotated[str | None, Field(description="Email (email) - Optional. The Child's email address. (max 100 characters; an email address)")] = None,
    address: Annotated[dict[str, Any] | None, Field(description="Address (address) - Optional. A street Address. Object keys: {line_1 (string): The first line of the Address; line_2 (string): The second line of the Address; line_3 (string): The third line of the Address; city (string): The city of the Address; county (string): The county of the Address; country (string): The country of the Address, in ISO3166 alpha 2 format; postcode (string): The postcode of the Address}")] = None,
    custom_fields: Annotated[list[dict[str, Any]] | None, Field(description="Custom fields (custom_fields) - Optional. Custom Field responses for this resource. HOW TO GET IT: Use children_list_custom_fields to find each field's `id` and type. Object keys: list of ONE of these shapes: {id (integer, required): The ID of the Custom Field; field_type (one of text/textarea/phone/date, required): A discriminator to determine the type of Custom Field this response belongs to; value (string, required): The raw value of a text, textarea, phone or date} OR {id (integer, required): The ID of the Custom Field; field_type (one of email, required): A discriminator to determine the type of Custom Field this response belongs to; value (string, required): The raw value of an email} OR {id (integer, required): The ID of the Custom Field; field_type (one of radio/select, required): A discriminator to determine the type of Custom Field this response belongs to; value ({id (string, required)}, required): The UUID of the selected response for a radio or select field} OR {id (integer, required): The ID of the Custom Field; field_type (one of checkbox, required): A discriminator to determine the type of Custom Field this response belongs to; value (list of {id (string, required)}, required): A list of UUIDs of the selected responses for a checkbox field}")] = None,
    communication: Annotated[dict[str, Any] | None, Field(description="Communication (communication) - Optional. The person's communication preferences. If no data is provided, the module default communication preferences will be used. Object keys: {general_email (boolean, required): Whether this person has consented to receiving general emails; general_sms (boolean, required): Whether this person has consented to receiving general SMS messages; phone (boolean, required): Whether this person has consented to receiving phone calls; post (boolean, required): Whether this person has consented to receiving post; rota_email (boolean, required): Whether this person has consented to receiving rota emails; rota_sms (boolean, required): Whether this person has consented to receiving rota SMS messages}")] = None,
    school: Annotated[str | None, Field(description="School (school) - Optional. The Child's school. (max 50 characters; cannot be empty)")] = None,
    school_year_offset: Annotated[int | None, Field(description="School year offset (school_year_offset) - Optional. Adjusted school year value based on the Child's age.")] = None,
    medical_short: Annotated[str | None, Field(description="Medical short (medical_short) - Optional. Condensed medical information about the Child. (max 100 characters; cannot be empty)")] = None,
    medical: Annotated[str | None, Field(description="Medical (medical) - Optional. Full medical information about the Child. (max 65535 characters; cannot be empty)")] = None,
    doctor_details: Annotated[str | None, Field(description="Doctor details (doctor_details) - Optional. Child's doctors details. (max 65535 characters; cannot be empty)")] = None,
    additional_needs: Annotated[str | None, Field(description="Additional needs (additional_needs) - Optional. Any additional needs declared about the Child. (max 65535 characters; cannot be empty)")] = None,
    additional_info: Annotated[str | None, Field(description="Additional info (additional_info) - Optional. Additional information about the Child. (max 250 characters; cannot be empty)")] = None,
    login: Annotated[dict[str, Any] | None, Field(description="Login (login) - Optional. My ChurchSuite/CharitySuite login information. Object keys: {enabled (boolean, required): Whether the person is able to log into My; invited (boolean, required): Whether the person has been invited to create a My account}")] = None,
    privacy: Annotated[dict[str, Any] | None, Field(description="Privacy (privacy) - Optional. The Child's privacy settings. If no data is provided, the default privacy settings will be used. Object keys: {name_visible (boolean, required): Whether the Child's name is visible in My; address_visible (boolean, required): Whether the Child's address is visible in My. This will only be respected if name_visible is true, else it will not be visible; mobile_visible (boolean, required): Whether the Child's mobile is visible in My. This will only be respected if name_visible is true, else it will not be visible; email_visible (boolean, required): Whether the Child's email is visible in My. This will only be respected if name_visible is true, else it will not be visible}")] = None,
) -> Any:
    """Create a child. Create a child with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - first_name (required): The first name of the Child (max 50 characters; cannot be empty)
      - last_name (required): The last name of the Child (max 50 characters; cannot be empty)
      - all_sites (required): Whether the Child belongs to All Sites
      - site_ids (required): An array of Site IDs that the Child belongs to. If `all_sites` is `true`, this will be overridden. You are only able to add site IDs for sites you have the relevant permission for -> No list tool for sites. Copy `site_ids` from an existing record (e.g. children_get_child), or use all_sites=true for every site.
      - middle_name (optional): The middle name of the Child (max 50 characters; cannot be empty)
      - formal_name (optional): The formal name of the Child (max 50 characters; cannot be empty)
      - sex (optional): The sex of the Child (one of: female, male)
      - date_of_birth (optional): The Child's date of birth (date as YYYY-MM-DD)
      - mobile (optional): The Child's mobile number (max 20 characters; cannot be empty)
      - email (optional): The Child's email address (max 100 characters; an email address)
      - address (optional): A street Address
      - custom_fields (optional): Custom Field responses for this resource -> Use children_list_custom_fields to find each field's `id` and type.
      - communication (optional): The person's communication preferences. If no data is provided, the module default communication preferences will be used
      - school (optional): The Child's school (max 50 characters; cannot be empty)
      - school_year_offset (optional): Adjusted school year value based on the Child's age
      - medical_short (optional): Condensed medical information about the Child (max 100 characters; cannot be empty)
      - medical (optional): Full medical information about the Child (max 65535 characters; cannot be empty)
      - doctor_details (optional): Child's doctors details (max 65535 characters; cannot be empty)
      - additional_needs (optional): Any additional needs declared about the Child (max 65535 characters; cannot be empty)
      - additional_info (optional): Additional information about the Child (max 250 characters; cannot be empty)
      - login (optional): My ChurchSuite/CharitySuite login information
      - privacy (optional): The Child's privacy settings. If no data is provided, the default privacy settings will be used"""
    body = _body({"first_name": first_name, "middle_name": middle_name, "last_name": last_name, "formal_name": formal_name, "sex": sex, "date_of_birth": date_of_birth, "mobile": mobile, "email": email, "address": address, "custom_fields": custom_fields, "communication": communication, "school": school, "school_year_offset": school_year_offset, "medical_short": medical_short, "medical": medical, "doctor_details": doctor_details, "additional_needs": additional_needs, "additional_info": additional_info, "login": login, "privacy": privacy, "all_sites": all_sites, "site_ids": site_ids}, None, (), keep=())
    return await _request("POST", "/children/children", None, body)

@tool("children", "POST")
async def children_create_child_note(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the child to add the note to. Find it with children_list_children (use `q` to search).")],
    content: Annotated[str, Field(description="Content (content) - REQUIRED. The content of the Note. (max 65535 characters; cannot be empty)")],
    pinned: Annotated[bool | None, Field(description="Pinned (pinned) - Optional. Whether the Note should be pinned.")] = None,
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - Optional. Object keys: ONE of these shapes: {just_me (one of true): The resource should only be visible to the current user} OR {user_group_ids (list of integer): List of user group IDs the resource is visible to. You are only able to add user group IDs for user groups you have access to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")] = None,
) -> Any:
    """Create a child note. Create a Note against a Child in the Children module.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the child to add the note to. Find it with children_list_children (use `q` to search).
      - content (required): The content of the Note (max 65535 characters; cannot be empty)
      - pinned (optional): Whether the Note should be pinned
      - visible_to (optional): Visible to"""
    body = _body({"content": content, "pinned": pinned, "visible_to": visible_to}, None, (), keep=())
    return await _request("POST", f"/children/children/{id}/notes", None, body)

@tool("children", "POST")
async def children_create_flow_stage(
    flow_id: Annotated[int, Field(description="Flow ID (flow_id) - REQUIRED. The Flow ID the Stage belongs to. HOW TO GET IT: Run children_list_flows and use the flow `id`.")],
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Stage's name. (max 50 characters; cannot be empty)")],
    due_interval: Annotated[int, Field(description="Due interval (due_interval) - REQUIRED. The number of days, weeks or months used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage.")],
    due_period: Annotated[Literal["days", "weeks", "months"], Field(description="Due period (due_period) - REQUIRED. The due period used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage. (one of: days, weeks, months)")],
    due_from: Annotated[Literal["first_added", "previous_stage"], Field(description="Due from (due_from) - REQUIRED. The relative point used for tracking a person linked to the Stage, either when they are first added to the flow or when they are processed from a previous stage. (one of: first_added, previous_stage)")],
    instructions: Annotated[str | None, Field(description="Instructions (instructions) - Optional. The instructions for the Stage. (max 65535 characters)")] = None,
    assigned_user_id: Annotated[int | None, Field(description="Assigned user ID (assigned_user_id) - Optional. The ID of the user automatically assigned to people in this stage, or null if no user is assigned. HOW TO GET IT: A ChurchSuite user ID. No list tool; it appears as `assigned_user_id` on existing flow stages/trackings.")] = None,
    automate_time: Annotated[str | None, Field(description="Automate time (automate_time) - Optional. The UTC time the Stage will automatically process any due or overdue people if it is automated, or null if it is not automated.")] = None,
) -> Any:
    """Create a flow stage. Create a stage for a flow.
    
    WHAT YOU NEED TO PROVIDE
      - flow_id (required): The Flow ID the Stage belongs to -> Run children_list_flows and use the flow `id`.
      - name (required): The Stage's name (max 50 characters; cannot be empty)
      - due_interval (required): The number of days, weeks or months used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage
      - due_period (required): The due period used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage (one of: days, weeks, months)
      - due_from (required): The relative point used for tracking a person linked to the Stage, either when they are first added to the flow or when they are processed from a previous stage (one of: first_added, previous_stage)
      - instructions (optional): The instructions for the Stage (max 65535 characters)
      - assigned_user_id (optional): The ID of the user automatically assigned to people in this stage, or null if no user is assigned -> A ChurchSuite user ID. No list tool; it appears as `assigned_user_id` on existing flow stages/trackings.
      - automate_time (optional): The UTC time the Stage will automatically process any due or overdue people if it is automated, or null if it is not automated"""
    body = _body({"flow_id": flow_id, "name": name, "instructions": instructions, "due_interval": due_interval, "due_period": due_period, "due_from": due_from, "assigned_user_id": assigned_user_id, "automate_time": automate_time}, None, (), keep=())
    return await _request("POST", "/children/flow_stages", None, body)

@tool("children", "POST")
async def children_create_flow_tracking(
    flow_id: Annotated[int, Field(description="Flow ID (flow_id) - REQUIRED. The Flow ID the Tracking is linked to. HOW TO GET IT: Run children_list_flows and use the flow `id`.")],
    stage_id: Annotated[int, Field(description="Stage ID (stage_id) - REQUIRED. The Stage ID the Tracking is linked to. HOW TO GET IT: Run children_list_flow_stages (filter flow_ids) and use the stage `id`.")],
    person: Annotated[dict[str, Any], Field(description="Person (person) - REQUIRED. The Tracking's Person details. Object keys: {type (string, required): The Person's type, indicating whether they are a Child, Contact, Customer or Giver -> For a child use 'children_child'.; id (integer, required): The Person's ID -> Child ID: run children_list_children (search with `q`) and use the `id`.}")],
) -> Any:
    """Create a flow tracking. Create a flow tracking, linked to a person's progress through a flow, with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - flow_id (required): The Flow ID the Tracking is linked to -> Run children_list_flows and use the flow `id`.
      - stage_id (required): The Stage ID the Tracking is linked to -> Run children_list_flow_stages (filter flow_ids) and use the stage `id`.
      - person (required): The Tracking's Person details"""
    body = _body({"flow_id": flow_id, "stage_id": stage_id, "person": person}, None, (), keep=())
    return await _request("POST", "/children/flow_trackings", None, body)

@tool("children", "POST")
async def children_create_key_date(
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Key Date's name. (max 50 characters; cannot be empty)")],
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], Field(description="Colour (colour) - REQUIRED. The Key Date's colour. (one of: blue, red, orange, green, purple, pink, brown, grey)")],
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - Optional. Object keys: ONE of these shapes: {just_me (one of true): The resource should only be visible to the current user} OR {user_group_ids (list of integer): List of user group IDs the resource is visible to. You are only able to add user group IDs for user groups you have access to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")] = None,
) -> Any:
    """Create a key date. Create a key date with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - name (required): The Key Date's name (max 50 characters; cannot be empty)
      - colour (required): The Key Date's colour (one of: blue, red, orange, green, purple, pink, brown, grey)
      - visible_to (optional): Visible to"""
    body = _body({"name": name, "colour": colour, "visible_to": visible_to}, None, (), keep=())
    return await _request("POST", "/children/key_dates", None, body)

@tool("children", "POST")
async def children_create_key_date_resource(
    key_date_id: Annotated[int, Field(description="Key date ID (key_date_id) - REQUIRED. The Key Date's ID. HOW TO GET IT: Run children_list_key_dates and use the key date `id`.")],
    resource: Annotated[dict[str, Any], Field(description="Resource (resource) - REQUIRED. The resource to link to the Key Date. Object keys: {id (integer, required): The ID of the resource -> Child ID: children_list_children.}")],
    date: Annotated[str, Field(description="Date (date) - REQUIRED. The date of the Key Date Resource. (date as YYYY-MM-DD)")],
    description: Annotated[str | None, Field(description="Description (description) - Optional. The description of this Key Date Resource. (max 65535 characters)")] = None,
) -> Any:
    """Create a key date resource. Create a key date resource adding a child to a key date
    
    WHAT YOU NEED TO PROVIDE
      - key_date_id (required): The Key Date's ID -> Run children_list_key_dates and use the key date `id`.
      - resource (required): The resource to link to the Key Date
      - date (required): The date of the Key Date Resource (date as YYYY-MM-DD)
      - description (optional): The description of this Key Date Resource (max 65535 characters)"""
    body = _body({"key_date_id": key_date_id, "resource": resource, "date": date, "description": description}, None, (), keep=())
    return await _request("POST", "/children/key_date_resources", None, body)

@tool("children", "POST")
async def children_create_tag(
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Tag's name. (max 50 characters; cannot be empty)")],
    is_smart: Annotated[bool, Field(description="Is smart (is_smart) - REQUIRED. Whether the Tag is smart or fixed.")],
    category_id: Annotated[int | None, Field(description="Category ID (category_id) - Optional. The Tag's Category ID. This sets the default colour of the Tag to the Category's colour. HOW TO GET IT: Run children_list_tag_categories and use the `id` of the category.")] = None,
    description: Annotated[str | None, Field(description="Description (description) - Optional. The description of the Tag. (max 65535 characters)")] = None,
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"] | None, Field(description="Colour (colour) - Optional. The name of the colour of the Tag. If not specified, the Category colour will be used where applicable, falling back to 'blue'. (one of: blue, red, orange, green, purple, pink, brown, grey)")] = None,
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - Optional. Object keys: ONE of these shapes: {just_me (one of true): The resource should only be visible to the current user} OR {user_group_ids (list of integer): List of user group IDs the resource is visible to. You are only able to add user group IDs for user groups you have access to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")] = None,
    pinned_icon: Annotated[Literal["tick", "cross", "archive", "warning", "flag", "magic", "star", "profile", "user", "couple", "group", "pledge", "education", "job", "gift", "role", "duration", "smiley-happy", "smiley-neutral", "smiley-sad"] | None, Field(description="Pinned icon (pinned_icon) - Optional. Pin this Tag by specifying an icon. Pinned tags give you a quick way to see your most important tags at a glance; they're displayed at the top of the page when viewing a person's profile. (one of: tick, cross, archive, warning, flag, magic, star, profile, user, couple, group, pledge, education, job, gift, role, duration, smiley-happy, smiley-neutral, smiley-sad)")] = None,
) -> Any:
    """Create a tag.
    
    WHAT YOU NEED TO PROVIDE
      - name (required): The Tag's name (max 50 characters; cannot be empty)
      - is_smart (required): Whether the Tag is smart or fixed
      - category_id (optional): The Tag's Category ID. This sets the default colour of the Tag to the Category's colour -> Run children_list_tag_categories and use the `id` of the category.
      - description (optional): The description of the Tag (max 65535 characters)
      - colour (optional): The name of the colour of the Tag. If not specified, the Category colour will be used where applicable, falling back to 'blue' (one of: blue, red, orange, green, purple, pink, brown, grey)
      - visible_to (optional): Visible to
      - pinned_icon (optional): Pin this Tag by specifying an icon. Pinned tags give you a quick way to see your most important tags at a glance; they're displayed at the top of the page when viewing a person's profile (one of: tick, cross, archive, warning, flag, magic, star, profile, user, couple, group, pledge, education, job, gift, role, duration, smiley-happy, smiley-neutral, smiley-sad)"""
    body = _body({"name": name, "category_id": category_id, "description": description, "colour": colour, "is_smart": is_smart, "visible_to": visible_to, "pinned_icon": pinned_icon}, None, (), keep=())
    return await _request("POST", "/children/tags", None, body)

@tool("children", "POST")
async def children_create_tag_resource(
    person: Annotated[dict[str, Any], Field(description="Person (person) - REQUIRED. The Person to link to the Tag. Object keys: {type (one of children_child, required): The Person's type, indicating whether they are a Child, Contact, Customer or Giver -> For a child use 'children_child'.; id (integer, required): The Person's ID -> Child ID: run children_list_children (search with `q`) and use the `id`.}")],
    tag_id: Annotated[int, Field(description="Tag ID (tag_id) - REQUIRED. The Tag's ID. HOW TO GET IT: Run children_list_tags and use the tag `id`.")],
) -> Any:
    """Create a tag resource. Create a tag resource linking a child to a tag
    
    WHAT YOU NEED TO PROVIDE
      - person (required): The Person to link to the Tag
      - tag_id (required): The Tag's ID -> Run children_list_tags and use the tag `id`."""
    body = _body({"person": person, "tag_id": tag_id}, None, (), keep=())
    return await _request("POST", "/children/tag_resources", None, body)

@tool("children", "PUT")
async def children_edit_child(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the child to change. Find it with children_list_children (use `q` to search) or the matching get tool.")],
    first_name: Annotated[str | None, Field(description="First name (first_name) - Optional - leave out to keep the current value. The first name of the Child. (max 50 characters; cannot be empty)")] = None,
    middle_name: Annotated[str | None, Field(description="Middle name (middle_name) - Optional - leave out to keep the current value. The middle name of the Child. (max 50 characters)")] = None,
    last_name: Annotated[str | None, Field(description="Last name (last_name) - Optional - leave out to keep the current value. The last name of the Child. (max 50 characters; cannot be empty)")] = None,
    formal_name: Annotated[str | None, Field(description="Formal name (formal_name) - Optional - leave out to keep the current value. The formal name of the Child. (max 50 characters)")] = None,
    sex: Annotated[Literal["female", "male"] | None, Field(description="Sex (sex) - Optional - leave out to keep the current value. The sex of the Child. (one of: female, male)")] = None,
    date_of_birth: Annotated[str | None, Field(description="Date of birth (date_of_birth) - Optional - leave out to keep the current value. The Child's date of birth. (date as YYYY-MM-DD)")] = None,
    mobile: Annotated[str | None, Field(description="Mobile (mobile) - Optional - leave out to keep the current value. The Child's mobile number. (max 20 characters)")] = None,
    email: Annotated[str | None, Field(description="Email (email) - Optional - leave out to keep the current value. The Child's email address. (max 100 characters; an email address)")] = None,
    address: Annotated[dict[str, Any] | None, Field(description="Address (address) - Optional - leave out to keep the current value. Edit a street Address. Object keys: {line_1 (string, required): The first line of the Address; line_2 (string, required): The second line of the Address; line_3 (string, required): The third line of the Address; city (string, required): The city of the Address; county (string, required): The county of the Address; country (string, required): The country of the Address, in ISO3166 alpha 2 format; postcode (string, required): The postcode of the Address}")] = None,
    custom_fields: Annotated[list[dict[str, Any]] | None, Field(description="Custom fields (custom_fields) - Optional - leave out to keep the current value. Custom Field responses for this resource. HOW TO GET IT: Use children_list_custom_fields to find each field's `id` and type. Object keys: list of ONE of these shapes: {id (integer, required): The ID of the Custom Field; field_type (one of text/textarea/phone/date, required): A discriminator to determine the type of Custom Field this response belongs to; value (string, required): The raw value of a text, textarea, phone or date} OR {id (integer, required): The ID of the Custom Field; field_type (one of email, required): A discriminator to determine the type of Custom Field this response belongs to; value (string, required): The raw value of an email} OR {id (integer, required): The ID of the Custom Field; field_type (one of radio/select, required): A discriminator to determine the type of Custom Field this response belongs to; value ({id (string, required)}, required): The UUID of the selected response for a radio or select field} OR {id (integer, required): The ID of the Custom Field; field_type (one of checkbox, required): A discriminator to determine the type of Custom Field this response belongs to; value (list of {id (string, required)}, required): A list of UUIDs of the selected responses for a checkbox field}")] = None,
    communication: Annotated[dict[str, Any] | None, Field(description="Communication (communication) - Optional - leave out to keep the current value. The person's communication preferences. Object keys: {general_email (boolean, required): Whether this person has consented to receiving general emails; general_sms (boolean, required): Whether this person has consented to receiving general SMS messages; phone (boolean, required): Whether this person has consented to receiving phone calls; post (boolean, required): Whether this person has consented to receiving post; rota_email (boolean, required): Whether this person has consented to receiving rota emails; rota_sms (boolean, required): Whether this person has consented to receiving rota SMS messages}")] = None,
    school: Annotated[str | None, Field(description="School (school) - Optional - leave out to keep the current value. The Child's school. (max 50 characters)")] = None,
    school_year_offset: Annotated[int | None, Field(description="School year offset (school_year_offset) - Optional - leave out to keep the current value. Adjusted school year value based on the Child's age.")] = None,
    medical_short: Annotated[str | None, Field(description="Medical short (medical_short) - Optional - leave out to keep the current value. Condensed medical information about the Child. (max 100 characters)")] = None,
    medical: Annotated[str | None, Field(description="Medical (medical) - Optional - leave out to keep the current value. Full medical information about the Child. (max 65535 characters)")] = None,
    doctor_details: Annotated[str | None, Field(description="Doctor details (doctor_details) - Optional - leave out to keep the current value. Child's doctors details. (max 65535 characters)")] = None,
    additional_needs: Annotated[str | None, Field(description="Additional needs (additional_needs) - Optional - leave out to keep the current value. Any additional needs declared about the Child. (max 65535 characters)")] = None,
    additional_info: Annotated[str | None, Field(description="Additional info (additional_info) - Optional - leave out to keep the current value. Additional information about the Child. (max 250 characters)")] = None,
    login: Annotated[dict[str, Any] | None, Field(description="Login (login) - Optional - leave out to keep the current value. My ChurchSuite/CharitySuite login information. Object keys: {enabled (boolean, required): Whether the person is able to log into My; invited (boolean, required): Whether the person has been invited to create a My account}")] = None,
    privacy: Annotated[dict[str, Any] | None, Field(description="Privacy (privacy) - Optional - leave out to keep the current value. The Child's privacy settings. Object keys: {name_visible (boolean, required): Whether the Child's name is visible in My; address_visible (boolean, required): Whether the Child's address is visible in My. This will only be respected if name_visible is true, else it will not be visible; mobile_visible (boolean, required): Whether the Child's mobile is visible in My. This will only be respected if name_visible is true, else it will not be visible; email_visible (boolean, required): Whether the Child's email is visible in My. This will only be respected if name_visible is true, else it will not be visible}")] = None,
    all_sites: Annotated[bool | None, Field(description="All sites (all_sites) - Optional - leave out to keep the current value. Whether the Child belongs to All Sites.")] = None,
    site_ids: Annotated[list[int] | None, Field(description="Site IDs (site_ids) - Optional - leave out to keep the current value. An array of Site IDs that the Child belongs to. If `all_sites` is `true`, this will be overridden. You are only able to add site IDs for sites you have the relevant permission for. HOW TO GET IT: No list tool for sites. Copy `site_ids` from an existing record (e.g. children_get_child), or use all_sites=true for every site.")] = None,
    set_null: Annotated[list[Literal["sex", "date_of_birth", "email", "address"]] | None, Field(description="Optional. Names of fields to CLEAR (set to null). Allowed: sex, date_of_birth, email, address. Leave a field out of both its own parameter and this list to keep it unchanged.")] = None,
) -> Any:
    """Update a child. Update a single child identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the child to change. Find it with children_list_children (use `q` to search) or the matching get tool.
      - first_name (optional): The first name of the Child (max 50 characters; cannot be empty)
      - middle_name (optional): The middle name of the Child (max 50 characters)
      - last_name (optional): The last name of the Child (max 50 characters; cannot be empty)
      - formal_name (optional): The formal name of the Child (max 50 characters)
      - sex (optional): The sex of the Child (one of: female, male)
      - date_of_birth (optional): The Child's date of birth (date as YYYY-MM-DD)
      - mobile (optional): The Child's mobile number (max 20 characters)
      - email (optional): The Child's email address (max 100 characters; an email address)
      - address (optional): Edit a street Address
      - custom_fields (optional): Custom Field responses for this resource -> Use children_list_custom_fields to find each field's `id` and type.
      - communication (optional): The person's communication preferences
      - school (optional): The Child's school (max 50 characters)
      - school_year_offset (optional): Adjusted school year value based on the Child's age
      - medical_short (optional): Condensed medical information about the Child (max 100 characters)
      - medical (optional): Full medical information about the Child (max 65535 characters)
      - doctor_details (optional): Child's doctors details (max 65535 characters)
      - additional_needs (optional): Any additional needs declared about the Child (max 65535 characters)
      - additional_info (optional): Additional information about the Child (max 250 characters)
      - login (optional): My ChurchSuite/CharitySuite login information
      - privacy (optional): The Child's privacy settings
      - all_sites (optional): Whether the Child belongs to All Sites
      - site_ids (optional): An array of Site IDs that the Child belongs to. If `all_sites` is `true`, this will be overridden. You are only able to add site IDs for sites you have the relevant permission for -> No list tool for sites. Copy `site_ids` from an existing record (e.g. children_get_child), or use all_sites=true for every site.
      - set_null (optional): list of fields to clear: sex, date_of_birth, email, address."""
    body = _body({"first_name": first_name, "middle_name": middle_name, "last_name": last_name, "formal_name": formal_name, "sex": sex, "date_of_birth": date_of_birth, "mobile": mobile, "email": email, "address": address, "custom_fields": custom_fields, "communication": communication, "school": school, "school_year_offset": school_year_offset, "medical_short": medical_short, "medical": medical, "doctor_details": doctor_details, "additional_needs": additional_needs, "additional_info": additional_info, "login": login, "privacy": privacy, "all_sites": all_sites, "site_ids": site_ids}, set_null, ("sex", "date_of_birth", "email", "address", ), keep=())
    return await _request("PUT", f"/children/children/{id}", None, body)

@tool("children", "PUT")
async def children_edit_flow_stage(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the flow stage to change. Find it with children_list_flow_stages (use `q` to search) or the matching get tool.")],
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Stage's name. (max 50 characters; cannot be empty)")],
    instructions: Annotated[str, Field(description="Instructions (instructions) - REQUIRED. The instructions for the Stage. (max 65535 characters)")],
    due_interval: Annotated[int, Field(description="Due interval (due_interval) - REQUIRED. The number of days, weeks or months used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage.")],
    due_period: Annotated[Literal["days", "weeks", "months"], Field(description="Due period (due_period) - REQUIRED. The due period used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage. (one of: days, weeks, months)")],
    due_from: Annotated[Literal["first_added", "previous_stage"], Field(description="Due from (due_from) - REQUIRED. The relative point used for tracking a person linked to the Stage, either when they are first added to the flow or when they are processed from a previous stage. (one of: first_added, previous_stage)")],
    assigned_user_id: Annotated[int | None, Field(description="Assigned user ID (assigned_user_id) - REQUIRED. The ID of the user automatically assigned to people in this stage, or null if no user is assigned. HOW TO GET IT: A ChurchSuite user ID. No list tool; it appears as `assigned_user_id` on existing flow stages/trackings.")],
    automate_time: Annotated[str | None, Field(description="Automate time (automate_time) - REQUIRED. The UTC time the Stage will automatically process any due or overdue people if it is automated, or null if it is not automated.")],
) -> Any:
    """Update a flow stage. Update a single stage identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the flow stage to change. Find it with children_list_flow_stages (use `q` to search) or the matching get tool.
      - name (required): The Stage's name (max 50 characters; cannot be empty)
      - instructions (required): The instructions for the Stage (max 65535 characters)
      - due_interval (required): The number of days, weeks or months used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage
      - due_period (required): The due period used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed from a previous stage (one of: days, weeks, months)
      - due_from (required): The relative point used for tracking a person linked to the Stage, either when they are first added to the flow or when they are processed from a previous stage (one of: first_added, previous_stage)
      - assigned_user_id (required): The ID of the user automatically assigned to people in this stage, or null if no user is assigned -> A ChurchSuite user ID. No list tool; it appears as `assigned_user_id` on existing flow stages/trackings.
      - automate_time (required): The UTC time the Stage will automatically process any due or overdue people if it is automated, or null if it is not automated"""
    body = _body({"name": name, "instructions": instructions, "due_interval": due_interval, "due_period": due_period, "due_from": due_from, "assigned_user_id": assigned_user_id, "automate_time": automate_time}, None, (), keep=("assigned_user_id", "automate_time", ))
    return await _request("PUT", f"/children/flow_stages/{id}", None, body)

@tool("children", "PUT")
async def children_edit_flow_tracking(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the flow tracking to change. Find it with children_list_flow_trackings (use `q` to search) or the matching get tool.")],
    due_date: Annotated[str, Field(description="Due date (due_date) - REQUIRED. The Tracking's due date. (date as YYYY-MM-DD)")],
    assigned_user_id: Annotated[int | None, Field(description="Assigned user ID (assigned_user_id) - REQUIRED. The Tracking's assigned user ID, or null if not assigned to a user. HOW TO GET IT: A ChurchSuite user ID. No list tool; it appears as `assigned_user_id` on existing flow stages/trackings.")],
) -> Any:
    """Update a flow tracking. Update a single flow tracking, linked to a person's progress through a flow, identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the flow tracking to change. Find it with children_list_flow_trackings (use `q` to search) or the matching get tool.
      - due_date (required): The Tracking's due date (date as YYYY-MM-DD)
      - assigned_user_id (required): The Tracking's assigned user ID, or null if not assigned to a user -> A ChurchSuite user ID. No list tool; it appears as `assigned_user_id` on existing flow stages/trackings."""
    body = _body({"due_date": due_date, "assigned_user_id": assigned_user_id}, None, (), keep=("assigned_user_id", ))
    return await _request("PUT", f"/children/flow_trackings/{id}", None, body)

@tool("children", "PUT")
async def children_edit_key_date(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the key date to change. Find it with children_list_key_dates (use `q` to search) or the matching get tool.")],
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Key Date's name. (max 50 characters; cannot be empty)")],
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], Field(description="Colour (colour) - REQUIRED. The Key Date's colour. (one of: blue, red, orange, green, purple, pink, brown, grey)")],
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - REQUIRED. Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is false, `user_group_ids` must contain at least one User Group ID. Object keys: ONE of these shapes: {just_me (one of false): The resource should only be visible to the current user; user_group_ids (list of integer, required): List of user group IDs the resource is visible to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.} OR {just_me (one of true, required): The resource should only be visible to the current user; user_group_ids (list of integer): List of user group IDs the resource is visible to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")],
) -> Any:
    """Update a key date. Update a single key date identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the key date to change. Find it with children_list_key_dates (use `q` to search) or the matching get tool.
      - name (required): The Key Date's name (max 50 characters; cannot be empty)
      - colour (required): The Key Date's colour (one of: blue, red, orange, green, purple, pink, brown, grey)
      - visible_to (required): Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is false, `user_group_ids` must contain at least one User Group ID"""
    body = _body({"name": name, "colour": colour, "visible_to": visible_to}, None, (), keep=("visible_to", ))
    return await _request("PUT", f"/children/key_dates/{id}", None, body)

@tool("children", "PUT")
async def children_edit_key_date_resource(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the key date resource to change. Find it with children_list_key_date_resources (use `q` to search) or the matching get tool.")],
    date: Annotated[str, Field(description="Date (date) - REQUIRED. The date of the Key Date Resource. (date as YYYY-MM-DD)")],
    description: Annotated[str, Field(description="Description (description) - REQUIRED. The description of this Key Date Resource. (max 65535 characters)")],
) -> Any:
    """Update a key date resource. Update a single key date resource identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the key date resource to change. Find it with children_list_key_date_resources (use `q` to search) or the matching get tool.
      - date (required): The date of the Key Date Resource (date as YYYY-MM-DD)
      - description (required): The description of this Key Date Resource (max 65535 characters)"""
    body = _body({"date": date, "description": description}, None, (), keep=())
    return await _request("PUT", f"/children/key_date_resources/{id}", None, body)

@tool("children", "PUT")
async def children_edit_tag(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the tag to change. Find it with children_list_tags (use `q` to search) or the matching get tool.")],
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Tag's name. (max 50 characters; cannot be empty)")],
    category_id: Annotated[int | None, Field(description="Category ID (category_id) - REQUIRED. The Tag's Category ID. HOW TO GET IT: Run children_list_tag_categories and use the `id` of the category.")],
    description: Annotated[str, Field(description="Description (description) - REQUIRED. The description of the Tag. (max 65535 characters)")],
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], Field(description="Colour (colour) - REQUIRED. The name of the colour of the Tag. (one of: blue, red, orange, green, purple, pink, brown, grey)")],
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - REQUIRED. Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is false, `user_group_ids` must contain at least one User Group ID. Object keys: ONE of these shapes: {just_me (one of false): The resource should only be visible to the current user; user_group_ids (list of integer, required): List of user group IDs the resource is visible to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.} OR {just_me (one of true, required): The resource should only be visible to the current user; user_group_ids (list of integer): List of user group IDs the resource is visible to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")],
    pinned_icon: Annotated[Literal["tick", "cross", "archive", "warning", "flag", "magic", "star", "profile", "user", "couple", "group", "pledge", "education", "job", "gift", "role", "duration", "smiley-happy", "smiley-neutral", "smiley-sad"] | None, Field(description="Pinned icon (pinned_icon) - REQUIRED. Pin this Tag by specifying an icon. This will be displayed within the application. (one of: tick, cross, archive, warning, flag, magic, star, profile, user, couple, group, pledge, education, job, gift, role, duration, smiley-happy, smiley-neutral, smiley-sad)")],
) -> Any:
    """Update a tag.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the tag to change. Find it with children_list_tags (use `q` to search) or the matching get tool.
      - name (required): The Tag's name (max 50 characters; cannot be empty)
      - category_id (required): The Tag's Category ID -> Run children_list_tag_categories and use the `id` of the category.
      - description (required): The description of the Tag (max 65535 characters)
      - colour (required): The name of the colour of the Tag (one of: blue, red, orange, green, purple, pink, brown, grey)
      - visible_to (required): Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is false, `user_group_ids` must contain at least one User Group ID
      - pinned_icon (required): Pin this Tag by specifying an icon. This will be displayed within the application (one of: tick, cross, archive, warning, flag, magic, star, profile, user, couple, group, pledge, education, job, gift, role, duration, smiley-happy, smiley-neutral, smiley-sad)"""
    body = _body({"name": name, "category_id": category_id, "description": description, "colour": colour, "visible_to": visible_to, "pinned_icon": pinned_icon}, None, (), keep=("category_id", "visible_to", "pinned_icon", ))
    return await _request("PUT", f"/children/tags/{id}", None, body)

@tool("children", "DELETE")
async def children_delete_child(id: int) -> Any:
    """DESTRUCTIVE. Delete a child. Delete a single child identified by ID."""
    return await _request("DELETE", f"/children/children/{id}", None)

@tool("children", "DELETE")
async def children_delete_flow_stage(id: int) -> Any:
    """DESTRUCTIVE. Delete a flow stage. Delete a single stage identified by ID. When a stage is deleted, all actions and flow trackings linked to the stage will be deleted."""
    return await _request("DELETE", f"/children/flow_stages/{id}", None)

@tool("children", "DELETE")
async def children_delete_flow_tracking(id: int) -> Any:
    """DESTRUCTIVE. Delete a flow tracking. Delete a single flow tracking, linked to a person's process through a flow, identified by ID."""
    return await _request("DELETE", f"/children/flow_trackings/{id}", None)

@tool("children", "DELETE")
async def children_delete_key_date(id: int) -> Any:
    """DESTRUCTIVE. Delete a key date. Delete a single key date identified by ID."""
    return await _request("DELETE", f"/children/key_dates/{id}", None)

@tool("children", "DELETE")
async def children_delete_key_date_resource(id: int) -> Any:
    """DESTRUCTIVE. Delete a key date resource. Delete a single key date resource identified by an ID, removing a child from a key date."""
    return await _request("DELETE", f"/children/key_date_resources/{id}", None)

@tool("children", "DELETE")
async def children_delete_tag(id: int) -> Any:
    """DESTRUCTIVE. Delete a tag. Delete a single tag identified by an ID."""
    return await _request("DELETE", f"/children/tags/{id}", None)

@tool("children", "DELETE")
async def children_delete_tag_resource(child_id: int, tag_id: int) -> Any:
    """DESTRUCTIVE. Delete a tag resource. Delete a single tag resource identified by a child ID and tag ID, unlinking a child from a tag.
    child_id (required): Filter records by the provided ID.
    tag_id (required): Filter records by the provided ID."""
    return await _request("DELETE", "/children/tag_resources", {"child_id": child_id, "tag_id": tag_id})



# ======== network ========

@tool("network", "GET")
async def network_get_custom_field(id: int) -> Any:
    """Get a custom field. Retrieve a single custom field identified by ID."""
    return await _request("GET", f"/network/custom_fields/{id}", None)

@tool("network", "GET")
async def network_get_key_date(id: int) -> Any:
    """Get a key date. Retrieve a single key date identified by ID."""
    return await _request("GET", f"/network/key_dates/{id}", None)

@tool("network", "GET")
async def network_get_label(id: int) -> Any:
    """Get a label. Retrieve a single label identified by ID."""
    return await _request("GET", f"/network/labels/{id}", None)

@tool("network", "GET")
async def network_get_organisation(id: int) -> Any:
    """Get an organisation. Retrieve a single organisation identified by ID."""
    return await _request("GET", f"/network/organisations/{id}", None)

@tool("network", "GET")
async def network_get_organisation_note(id: int) -> Any:
    """List organisation notes. Returns a list of organisation notes, sorted by pinned, created_at."""
    return await _request("GET", f"/network/organisations/{id}/notes", None)

@tool("network", "GET")
async def network_get_role(id: int) -> Any:
    """Get a role. Retrieve a single role identified by ID."""
    return await _request("GET", f"/network/roles/{id}", None)

@tool("network", "GET")
async def network_list_custom_fields(page: int | None = None, per_page: int | None = None) -> Any:
    """List custom fields. Returns a list of custom fields"""
    return await _request("GET", "/network/custom_fields", {"page": page, "per_page": per_page})

@tool("network", "GET")
async def network_list_key_date_resources(page: int | None = None, per_page: int | None = None, key_date_ids: list[int] | None = None, organisation_ids: list[int] | None = None, order_by: Literal["created_at:asc", "created_at:desc", "date:asc", "date:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc"] | None = None) -> Any:
    """List key date resources. Returns a list of key date resources, optionally filtered by key date or organisation ID. A key date resource records an instance of a key date for an organisation on a given date.
    key_date_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    organisation_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/network/key_date_resources", {"page": page, "per_page": per_page, "key_date_ids[]": key_date_ids, "organisation_ids[]": organisation_ids, "order_by": order_by})

@tool("network", "GET")
async def network_list_key_dates(page: int | None = None, per_page: int | None = None, q: str | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List key dates. Returns a list of key dates, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/network/key_dates", {"page": page, "per_page": per_page, "q": q, "order_by": order_by})

@tool("network", "GET")
async def network_list_label_resources(page: int | None = None, per_page: int | None = None, label_ids: list[int] | None = None, organisation_ids: list[int] | None = None) -> Any:
    """List label resources. Returns a list of label resources
    label_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    organisation_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/network/label_resources", {"page": page, "per_page": per_page, "label_ids[]": label_ids, "organisation_ids[]": organisation_ids})

@tool("network", "GET")
async def network_list_labels(page: int | None = None, per_page: int | None = None, q: str | None = None) -> Any:
    """List labels. Returns a list of labels, sorted by their order property.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/network/labels", {"page": page, "per_page": per_page, "q": q})

@tool("network", "GET")
async def network_list_members(page: int | None = None, per_page: int | None = None, role_ids: list[int] | None = None, organisation_ids: list[int] | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List members. List organisation members.
    role_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    organisation_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/network/members", {"page": page, "per_page": per_page, "role_ids[]": role_ids, "organisation_ids[]": organisation_ids, "order_by": order_by})

@tool("network", "GET")
async def network_list_organisations(page: int | None = None, per_page: int | None = None, site_ids: list[int] | None = None, q: str | None = None, status: Literal["active", "archived"] | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List organisations. Returns a list of organisation, sorted alphabetically by name.
    site_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    q: Filter records with a fuzzy string match against the following properties: name, reference (if enabled), meeting address city.
    status: Filter records by the provided status.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/network/organisations", {"page": page, "per_page": per_page, "site_ids[]": site_ids, "q": q, "status": status, "order_by": order_by})

@tool("network", "GET")
async def network_list_roles(page: int | None = None, per_page: int | None = None, q: str | None = None) -> Any:
    """List roles. Returns a list of roles, sorted by their order property.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/network/roles", {"page": page, "per_page": per_page, "q": q})

@tool("network", "POST")
async def network_create_key_date(
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Key Date's name. (max 50 characters; cannot be empty)")],
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], Field(description="Colour (colour) - REQUIRED. The Key Date's colour. (one of: blue, red, orange, green, purple, pink, brown, grey)")],
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - Optional. Object keys: ONE of these shapes: {just_me (one of true): The resource should only be visible to the current user} OR {user_group_ids (list of integer): List of user group IDs the resource is visible to. You are only able to add user group IDs for user groups you have access to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")] = None,
) -> Any:
    """Create a key date. Create a key date with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - name (required): The Key Date's name (max 50 characters; cannot be empty)
      - colour (required): The Key Date's colour (one of: blue, red, orange, green, purple, pink, brown, grey)
      - visible_to (optional): Visible to"""
    body = _body({"name": name, "colour": colour, "visible_to": visible_to}, None, (), keep=())
    return await _request("POST", "/network/key_dates", None, body)

@tool("network", "POST")
async def network_create_key_date_resource(
    key_date_id: Annotated[int, Field(description="Key date ID (key_date_id) - REQUIRED. The Key Date's ID. HOW TO GET IT: Run network_list_key_dates and use the key date `id`.")],
    resource: Annotated[dict[str, Any], Field(description="Resource (resource) - REQUIRED. The resource to link to the Key Date. Object keys: {id (integer, required): The ID of the resource -> Organisation ID: network_list_organisations.}")],
    date: Annotated[str, Field(description="Date (date) - REQUIRED. The date of the Key Date Resource. (date as YYYY-MM-DD)")],
    description: Annotated[str | None, Field(description="Description (description) - Optional. The description of this Key Date Resource. (max 65535 characters)")] = None,
) -> Any:
    """Create a key date resource. Create a key date resource adding an organisation to a key date
    
    WHAT YOU NEED TO PROVIDE
      - key_date_id (required): The Key Date's ID -> Run network_list_key_dates and use the key date `id`.
      - resource (required): The resource to link to the Key Date
      - date (required): The date of the Key Date Resource (date as YYYY-MM-DD)
      - description (optional): The description of this Key Date Resource (max 65535 characters)"""
    body = _body({"key_date_id": key_date_id, "resource": resource, "date": date, "description": description}, None, (), keep=())
    return await _request("POST", "/network/key_date_resources", None, body)

@tool("network", "POST")
async def network_create_organisation_note(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the organisation to add the note to. Find it with network_list_organisations (use `q` to search).")],
    content: Annotated[str, Field(description="Content (content) - REQUIRED. The content of the Note. (max 65535 characters; cannot be empty)")],
    pinned: Annotated[bool | None, Field(description="Pinned (pinned) - Optional. Whether the Note should be pinned.")] = None,
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - Optional. Object keys: ONE of these shapes: {just_me (one of true): The resource should only be visible to the current user} OR {user_group_ids (list of integer): List of user group IDs the resource is visible to. You are only able to add user group IDs for user groups you have access to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")] = None,
) -> Any:
    """Create a organisation note. Create a Note against a Organisation in the Network module.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the organisation to add the note to. Find it with network_list_organisations (use `q` to search).
      - content (required): The content of the Note (max 65535 characters; cannot be empty)
      - pinned (optional): Whether the Note should be pinned
      - visible_to (optional): Visible to"""
    body = _body({"content": content, "pinned": pinned, "visible_to": visible_to}, None, (), keep=())
    return await _request("POST", f"/network/organisations/{id}/notes", None, body)

@tool("network", "PUT")
async def network_edit_key_date(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the key date to change. Find it with network_list_key_dates (use `q` to search) or the matching get tool.")],
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Key Date's name. (max 50 characters; cannot be empty)")],
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], Field(description="Colour (colour) - REQUIRED. The Key Date's colour. (one of: blue, red, orange, green, purple, pink, brown, grey)")],
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - REQUIRED. Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is false, `user_group_ids` must contain at least one User Group ID. Object keys: ONE of these shapes: {just_me (one of false): The resource should only be visible to the current user; user_group_ids (list of integer, required): List of user group IDs the resource is visible to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.} OR {just_me (one of true, required): The resource should only be visible to the current user; user_group_ids (list of integer): List of user group IDs the resource is visible to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")],
) -> Any:
    """Update a key date. Update a single key date identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the key date to change. Find it with network_list_key_dates (use `q` to search) or the matching get tool.
      - name (required): The Key Date's name (max 50 characters; cannot be empty)
      - colour (required): The Key Date's colour (one of: blue, red, orange, green, purple, pink, brown, grey)
      - visible_to (required): Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is false, `user_group_ids` must contain at least one User Group ID"""
    body = _body({"name": name, "colour": colour, "visible_to": visible_to}, None, (), keep=("visible_to", ))
    return await _request("PUT", f"/network/key_dates/{id}", None, body)

@tool("network", "PUT")
async def network_edit_key_date_resource(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the key date resource to change. Find it with network_list_key_date_resources (use `q` to search) or the matching get tool.")],
    date: Annotated[str, Field(description="Date (date) - REQUIRED. The date of the Key Date Resource. (date as YYYY-MM-DD)")],
    description: Annotated[str, Field(description="Description (description) - REQUIRED. The description of this Key Date Resource. (max 65535 characters)")],
) -> Any:
    """Update a key date resource. Update a single key date resource identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the key date resource to change. Find it with network_list_key_date_resources (use `q` to search) or the matching get tool.
      - date (required): The date of the Key Date Resource (date as YYYY-MM-DD)
      - description (required): The description of this Key Date Resource (max 65535 characters)"""
    body = _body({"date": date, "description": description}, None, (), keep=())
    return await _request("PUT", f"/network/key_date_resources/{id}", None, body)

@tool("network", "DELETE")
async def network_delete_key_date(id: int) -> Any:
    """DESTRUCTIVE. Delete a key date. Delete a single key date identified by ID."""
    return await _request("DELETE", f"/network/key_dates/{id}", None)

@tool("network", "DELETE")
async def network_delete_key_date_resource(id: int) -> Any:
    """DESTRUCTIVE. Delete a key date resource. Delete a single key date resource identified by an ID, removing an organisation from a key date."""
    return await _request("DELETE", f"/network/key_date_resources/{id}", None)

@tool("network", "POST")
async def network_create_label(
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Label's name. (max 50 characters; cannot be empty)")],
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], Field(description="Colour (colour) - REQUIRED. The Label's colour. (one of: blue, red, orange, green, purple, pink, brown, grey)")],
    options: Annotated[list[str], Field(description="Options (options) - REQUIRED. List of options that belong to the Label.")],
    multiple_responses: Annotated[bool, Field(description="Multiple responses (multiple_responses) - REQUIRED. Whether the Label can have multiple responses.")],
    required: Annotated[bool, Field(description="Required (required) - REQUIRED. Whether the Label is required.")],
) -> Any:
    """Create a label. Create a label with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - name (required): The Label's name (max 50 characters; cannot be empty)
      - colour (required): The Label's colour (one of: blue, red, orange, green, purple, pink, brown, grey)
      - options (required): List of options that belong to the Label
      - multiple_responses (required): Whether the Label can have multiple responses
      - required (required): Whether the Label is required"""
    body = _body({"name": name, "colour": colour, "options": options, "multiple_responses": multiple_responses, "required": required}, None, (), keep=())
    return await _request("POST", "/network/labels", None, body)

@tool("network", "PUT")
async def network_edit_label(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the label to change. Find it with network_list_labels (use `q` to search) or the matching get tool.")],
    name: Annotated[str | None, Field(description="Name (name) - Optional - leave out to keep the current value. The Label's name. (max 50 characters; cannot be empty)")] = None,
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"] | None, Field(description="Colour (colour) - Optional - leave out to keep the current value. The Label's colour. (one of: blue, red, orange, green, purple, pink, brown, grey)")] = None,
    multiple_responses: Annotated[bool | None, Field(description="Multiple responses (multiple_responses) - Optional - leave out to keep the current value. Whether the Label can have multiple responses.")] = None,
    required: Annotated[bool | None, Field(description="Required (required) - Optional - leave out to keep the current value. Whether the Label is required.")] = None,
) -> Any:
    """Update a label. Update a single label identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the label to change. Find it with network_list_labels (use `q` to search) or the matching get tool.
      - name (optional): The Label's name (max 50 characters; cannot be empty)
      - colour (optional): The Label's colour (one of: blue, red, orange, green, purple, pink, brown, grey)
      - multiple_responses (optional): Whether the Label can have multiple responses
      - required (optional): Whether the Label is required"""
    body = _body({"name": name, "colour": colour, "multiple_responses": multiple_responses, "required": required}, None, (), keep=())
    return await _request("PUT", f"/network/labels/{id}", None, body)

@tool("network", "DELETE")
async def network_delete_label(id: int) -> Any:
    """DESTRUCTIVE. Delete a label. Delete a single label identified by ID. Label responses will be deleted from all organisations. Module settings that reference this label will be affected."""
    return await _request("DELETE", f"/network/labels/{id}", None)



# ======== planning ========

@tool("planning", "GET")
async def planning_get_library_item(id: int) -> Any:
    """Get a Library Item. Retrieve a single library item identified by ID."""
    return await _request("GET", f"/planning/library_items/{id}", None)

@tool("planning", "GET")
async def planning_get_plan(id: int) -> Any:
    """Get a plan. Retrieve a single plan identified by ID."""
    return await _request("GET", f"/planning/plans/{id}", None)

@tool("planning", "GET")
async def planning_get_plan_item(id: int) -> Any:
    """Get a plan item. Retrieve a single plan item identified by ID."""
    return await _request("GET", f"/planning/plan_items/{id}", None)

@tool("planning", "GET")
async def planning_get_plan_note(id: int) -> Any:
    """List plan notes. Returns a list of plan notes, sorted by pinned, created_at."""
    return await _request("GET", f"/planning/plans/{id}/notes", None)

@tool("planning", "GET")
async def planning_get_song(id: int) -> Any:
    """Retrieve a song. Retrieve a single song, identified by ID."""
    return await _request("GET", f"/planning/songs/{id}", None)

@tool("planning", "GET")
async def planning_get_song_arrangement(id: int) -> Any:
    """Retrieve a Song Arrangement. Retrieve a single song arrangement identified by ID."""
    return await _request("GET", f"/planning/song_arrangements/{id}", None)

@tool("planning", "GET")
async def planning_get_template(id: int) -> Any:
    """Get a template. Retrieve a single template identified by ID."""
    return await _request("GET", f"/planning/templates/{id}", None)

@tool("planning", "GET")
async def planning_get_template_item(id: int) -> Any:
    """Get a template item. Retrieve a single template item identified by ID."""
    return await _request("GET", f"/planning/template_items/{id}", None)

@tool("planning", "GET")
async def planning_get_type(id: int) -> Any:
    """Get a type. Retrieve a single type identified by ID."""
    return await _request("GET", f"/planning/types/{id}", None)

@tool("planning", "GET")
async def planning_list_library_items(page: int | None = None, per_page: int | None = None, q: str | None = None, status: Literal["active", "archived"] | None = None, type_ids: list[int] | None = None) -> Any:
    """List library items. Returns a list of library items, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name.
    status: Filter records by the provided status.
    type_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/planning/library_items", {"page": page, "per_page": per_page, "q": q, "status": status, "type_ids[]": type_ids})

@tool("planning", "GET")
async def planning_list_plan_items(plan_ids: list[int], page: int | None = None, per_page: int | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc", "order:asc", "order:desc"] | None = None) -> Any:
    """List plan items. Returns a list of plan items, sorted by ID.
    plan_ids (required): Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/planning/plan_items", {"plan_ids[]": plan_ids, "page": page, "per_page": per_page, "order_by": order_by})

@tool("planning", "GET")
async def planning_list_plans(page: int | None = None, per_page: int | None = None, site_ids: list[int] | None = None, q: str | None = None, starts_after: str | None = None, starts_before: str | None = None, status: Literal["draft", "published"] | None = None, order_by: Literal["created_at:asc", "created_at:desc", "date:asc", "date:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List plans. Returns a list of plans, sorted by date and time.
    site_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    q: Filter records with a fuzzy string match against the following properties: name.
    starts_after: Only include records that start after this date.
    starts_before: Only include records that start before this date.
    status: Filter records by the provided status.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/planning/plans", {"page": page, "per_page": per_page, "site_ids[]": site_ids, "q": q, "starts_after": starts_after, "starts_before": starts_before, "status": status, "order_by": order_by})

@tool("planning", "GET")
async def planning_list_song_arrangements(page: int | None = None, per_page: int | None = None, q: str | None = None, song_ids: list[int] | None = None, song_statuses: list[Literal["active", "archived"]] | None = None) -> Any:
    """List song arrangements. Returns a list of song arrangements, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name, artist, song_ccli, song_name.
    song_ids: Filter Song Arrangements by Song IDs.
    song_statuses: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/planning/song_arrangements", {"page": page, "per_page": per_page, "q": q, "song_ids[]": song_ids, "song_statuses[]": song_statuses})

@tool("planning", "GET")
async def planning_list_songs(page: int | None = None, per_page: int | None = None, last_used_after: str | None = None, q: str | None = None, statuses: list[Literal["active", "archived"]] | None = None) -> Any:
    """List songs. Returns a list of Songs, sorted alphabetically.
    last_used_after: Only include records that were used after this date.
    q: Filter records with a fuzzy string match against the following properties: name, ccli_number, copyright, administration.
    statuses: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/planning/songs", {"page": page, "per_page": per_page, "last_used_after": last_used_after, "q": q, "statuses[]": statuses})

@tool("planning", "GET")
async def planning_list_template_items(page: int | None = None, per_page: int | None = None, arrangement_ids: list[int] | None = None, library_item_ids: list[int] | None = None, template_ids: list[int] | None = None) -> Any:
    """List template items. Returns a list of template items, sorted by date.
    arrangement_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    library_item_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    template_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/planning/template_items", {"page": page, "per_page": per_page, "arrangement_ids[]": arrangement_ids, "library_item_ids[]": library_item_ids, "template_ids[]": template_ids})

@tool("planning", "GET")
async def planning_list_templates(page: int | None = None, per_page: int | None = None, q: str | None = None, status: Literal["active", "archived"] | None = None) -> Any:
    """List templates. Returns a list of templates, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name.
    status: Filter records by the provided status."""
    return await _request("GET", "/planning/templates", {"page": page, "per_page": per_page, "q": q, "status": status})

@tool("planning", "GET")
async def planning_list_types(page: int | None = None, per_page: int | None = None, q: str | None = None) -> Any:
    """List types. Returns a list of types, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/planning/types", {"page": page, "per_page": per_page, "q": q})

@tool("planning", "POST")
async def planning_create_plan_note(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the plan to add the note to. Find it with planning_list_plans (use `q` to search).")],
    content: Annotated[str, Field(description="Content (content) - REQUIRED. The content of the Note. (max 65535 characters; cannot be empty)")],
    pinned: Annotated[bool | None, Field(description="Pinned (pinned) - Optional. Whether the Note should be pinned.")] = None,
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - Optional. Object keys: ONE of these shapes: {just_me (one of true): The resource should only be visible to the current user} OR {user_group_ids (list of integer): List of user group IDs the resource is visible to. You are only able to add user group IDs for user groups you have access to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")] = None,
) -> Any:
    """Create a plan note. Create a Note against a Plan in the Planning module.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the plan to add the note to. Find it with planning_list_plans (use `q` to search).
      - content (required): The content of the Note (max 65535 characters; cannot be empty)
      - pinned (optional): Whether the Note should be pinned
      - visible_to (optional): Visible to"""
    body = _body({"content": content, "pinned": pinned, "visible_to": visible_to}, None, (), keep=())
    return await _request("POST", f"/planning/plans/{id}/notes", None, body)



# ======== rotas ========

@tool("rotas", "GET")
async def rotas_get_ministry(id: int) -> Any:
    """Get a ministry. Retrieve a single ministry identified by ID."""
    return await _request("GET", f"/rotas/ministries/{id}", None)

@tool("rotas", "GET")
async def rotas_get_ministry_team(id: int) -> Any:
    """Get a ministry team. Retrieve a single ministry team identified by ID."""
    return await _request("GET", f"/rotas/ministry_teams/{id}", None)

@tool("rotas", "GET")
async def rotas_list_ministries(page: int | None = None, per_page: int | None = None, site_ids: list[int] | None = None, statuses: list[Literal["active", "archived"]] | None = None, q: str | None = None, ids: list[int] | None = None) -> Any:
    """List ministries. Returns a list of ministries, sorted by date created.
    site_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    statuses: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    q: Filter records with a fuzzy string match against the following properties: name.
    ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/rotas/ministries", {"page": page, "per_page": per_page, "site_ids[]": site_ids, "statuses[]": statuses, "q": q, "ids[]": ids})

@tool("rotas", "GET")
async def rotas_list_ministry_members(page: int | None = None, per_page: int | None = None, ministry_ids: list[int] | None = None, role_ids: list[int] | None = None, team_ids: list[int] | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List ministry members. Returns a list of team members.
    ministry_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    role_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    team_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/rotas/ministry_members", {"page": page, "per_page": per_page, "ministry_ids[]": ministry_ids, "role_ids[]": role_ids, "team_ids[]": team_ids, "order_by": order_by})

@tool("rotas", "GET")
async def rotas_list_ministry_teams(page: int | None = None, per_page: int | None = None, ministry_ids: list[int] | None = None, team_ids: list[int] | None = None) -> Any:
    """List ministry teams. Returns a list of ministry teams, sorted by date created.
    ministry_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    team_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/rotas/ministry_teams", {"page": page, "per_page": per_page, "ministry_ids[]": ministry_ids, "team_ids[]": team_ids})

@tool("rotas", "GET")
async def rotas_list_roles(page: int | None = None, per_page: int | None = None, member_ids: list[int] | None = None, ministry_ids: list[int] | None = None) -> Any:
    """List roles. Returns a list of roles, sorted by date created.
    member_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    ministry_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/rotas/roles", {"page": page, "per_page": per_page, "member_ids[]": member_ids, "ministry_ids[]": ministry_ids})

@tool("rotas", "GET")
async def rotas_list_unavailability(page: int | None = None, per_page: int | None = None, contact_ids: list[int] | None = None, child_ids: list[int] | None = None, ministry_ids: list[int] | None = None, starts_before: str | None = None, starts_after: str | None = None, order_by: Literal["created_at:asc", "created_at:desc", "date:asc", "date:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc"] | None = None) -> Any:
    """List unavailability. Returns a list of unavailability, sorted by date created.
    contact_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    child_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    ministry_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    starts_before: Only include records that start before this date.
    starts_after: Only include records that start after this date.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/rotas/unavailability", {"page": page, "per_page": per_page, "contact_ids[]": contact_ids, "child_ids[]": child_ids, "ministry_ids[]": ministry_ids, "starts_before": starts_before, "starts_after": starts_after, "order_by": order_by})

@tool("rotas", "POST")
async def rotas_create_unavailability(
    person: Annotated[dict[str, Any], Field(description="Person (person) - REQUIRED. The Person this Unavailability record belongs to. Object keys: {type (one of addressbook_contact/children_child, required): The Person's type, indicating whether they are a Contact, or Child -> 'addressbook_contact' for a Contact or 'children_child' for a Child.; id (integer, required): The Person's ID -> A Contact ID (addressbook_list_contacts) or Child ID (children_list_children), matching `type`.}")],
    starts_at: Annotated[str, Field(description="Starts at (starts_at) - REQUIRED. The date and time the Unavailability starts. To create unavailability lasting all day, start at 00:00:00 and end at 23:59:59. (ISO 8601 date-time; a trailing Z means UTC, e.g. 2026-11-01T10:00:00Z)")],
    ends_at: Annotated[str, Field(description="Ends at (ends_at) - REQUIRED. The date and time the Unavailability ends. To create unavailability lasting all day, start at 00:00:00 and end at 23:59:59. (ISO 8601 date-time; a trailing Z means UTC, e.g. 2026-11-01T10:00:00Z)")],
    description: Annotated[str | None, Field(description="Description (description) - Optional. The Unavailability's description. (max 250 characters; cannot be empty)")] = None,
) -> Any:
    """Create unavailability. Create unavailability, with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - person (required): The Person this Unavailability record belongs to
      - starts_at (required): The date and time the Unavailability starts. To create unavailability lasting all day, start at 00:00:00 and end at 23:59:59 (ISO 8601 date-time; a trailing Z means UTC, e.g. 2026-11-01T10:00:00Z)
      - ends_at (required): The date and time the Unavailability ends. To create unavailability lasting all day, start at 00:00:00 and end at 23:59:59 (ISO 8601 date-time; a trailing Z means UTC, e.g. 2026-11-01T10:00:00Z)
      - description (optional): The Unavailability's description (max 250 characters; cannot be empty)"""
    body = _body({"person": person, "description": description, "starts_at": starts_at, "ends_at": ends_at}, None, (), keep=())
    return await _request("POST", "/rotas/unavailability", None, body)

@tool("rotas", "DELETE")
async def rotas_delete_unavailability(id: int) -> Any:
    """DESTRUCTIVE. Delete an unavailability. Delete a single unavailability, identified by ID."""
    return await _request("DELETE", f"/rotas/unavailability/{id}", None)



# ======== smallgroups ========

@tool("smallgroups", "GET")
async def smallgroups_get_cluster(id: int) -> Any:
    """Get a cluster. Retrieve a single cluster identified by ID."""
    return await _request("GET", f"/smallgroups/clusters/{id}", None)

@tool("smallgroups", "GET")
async def smallgroups_get_custom_field(id: int) -> Any:
    """Get a custom field. Retrieve a single custom field identified by ID."""
    return await _request("GET", f"/smallgroups/custom_fields/{id}", None)

@tool("smallgroups", "GET")
async def smallgroups_get_group(id: int) -> Any:
    """Get a group. Retrieve a single group identified by ID."""
    return await _request("GET", f"/smallgroups/groups/{id}", None)

@tool("smallgroups", "GET")
async def smallgroups_get_group_note(id: int) -> Any:
    """List group notes. Returns a list of group notes, sorted by pinned, created_at."""
    return await _request("GET", f"/smallgroups/groups/{id}/notes", None)

@tool("smallgroups", "GET")
async def smallgroups_get_label(id: int) -> Any:
    """Get a label. Retrieve a single label identified by ID."""
    return await _request("GET", f"/smallgroups/labels/{id}", None)

@tool("smallgroups", "GET")
async def smallgroups_get_member(id: int) -> Any:
    """Get a member. Retrieve a single member identified by ID."""
    return await _request("GET", f"/smallgroups/members/{id}", None)

@tool("smallgroups", "GET")
async def smallgroups_get_role(id: int) -> Any:
    """Get a role. Retrieve a single role identified by ID."""
    return await _request("GET", f"/smallgroups/roles/{id}", None)

@tool("smallgroups", "GET")
async def smallgroups_list_attendances(page: int | None = None, per_page: int | None = None, group_ids: list[int] | None = None, before: str | None = None, after: str | None = None) -> Any:
    """List attendance. Returns a list of attendance.
    group_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    before: Only include records before this date.
    after: Only include records after this date."""
    return await _request("GET", "/smallgroups/attendances", {"page": page, "per_page": per_page, "group_ids[]": group_ids, "before": before, "after": after})

@tool("smallgroups", "GET")
async def smallgroups_list_clusters(page: int | None = None, per_page: int | None = None, q: str | None = None) -> Any:
    """List clusters. Returns a list of clusters, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/smallgroups/clusters", {"page": page, "per_page": per_page, "q": q})

@tool("smallgroups", "GET")
async def smallgroups_list_custom_fields(page: int | None = None, per_page: int | None = None) -> Any:
    """List custom fields. Returns a list of custom fields"""
    return await _request("GET", "/smallgroups/custom_fields", {"page": page, "per_page": per_page})

@tool("smallgroups", "GET")
async def smallgroups_list_groups(page: int | None = None, per_page: int | None = None, site_ids: list[int] | None = None, cluster_ids: list[int] | None = None, frequency: Literal["weekly", "fortnightly", "monthly", "quarterly", "custom"] | None = None, days: list[Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]] | None = None, q: str | None = None, view: Literal["past", "active", "active_future", "future"] | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List groups. Returns a list of groups, sorted alphabetically by name.
    site_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    cluster_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    frequency: Filter records by the provided value.
    days: Filter records by the provided value.
    q: Filter records with a fuzzy string match against the following properties: name, location.
    view: Filter records to view only those that have a date range that is in the past, is active or is in the future.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/smallgroups/groups", {"page": page, "per_page": per_page, "site_ids[]": site_ids, "cluster_ids[]": cluster_ids, "frequency": frequency, "days[]": days, "q": q, "view": view, "order_by": order_by})

@tool("smallgroups", "GET")
async def smallgroups_list_label_resources(page: int | None = None, per_page: int | None = None, label_ids: list[int] | None = None, group_ids: list[int] | None = None) -> Any:
    """List label resources. Returns a list of label resources
    label_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    group_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/smallgroups/label_resources", {"page": page, "per_page": per_page, "label_ids[]": label_ids, "group_ids[]": group_ids})

@tool("smallgroups", "GET")
async def smallgroups_list_labels(page: int | None = None, per_page: int | None = None, q: str | None = None) -> Any:
    """List labels. Returns a list of labels, sorted by their order property.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/smallgroups/labels", {"page": page, "per_page": per_page, "q": q})

@tool("smallgroups", "GET")
async def smallgroups_list_members(page: int | None = None, per_page: int | None = None, child_ids: list[int] | None = None, contact_ids: list[int] | None = None, role_id: int | None = None, group_ids: list[int] | None = None, status: Literal["active", "pending"] | None = None, order_by: Literal["created_at:asc", "created_at:desc", "id:asc", "id:desc", "modified_at:asc", "modified_at:desc", "name:asc", "name:desc"] | None = None) -> Any:
    """List members. Returns a list of members, sorted alphabetically by name.
    child_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    contact_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    role_id: Filter records by the provided value.
    group_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    status: Filter records by the provided status.
    order_by: Order records by the specified field and sort direction."""
    return await _request("GET", "/smallgroups/members", {"page": page, "per_page": per_page, "child_ids[]": child_ids, "contact_ids[]": contact_ids, "role_id": role_id, "group_ids[]": group_ids, "status": status, "order_by": order_by})

@tool("smallgroups", "GET")
async def smallgroups_list_roles(page: int | None = None, per_page: int | None = None) -> Any:
    """List roles. Returns a list of roles, sorted by their order property."""
    return await _request("GET", "/smallgroups/roles", {"page": page, "per_page": per_page})

@tool("smallgroups", "POST")
async def smallgroups_create_cluster(
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Cluster's name. (max 50 characters; cannot be empty)")],
) -> Any:
    """Create a cluster. Create a cluster with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - name (required): The Cluster's name (max 50 characters; cannot be empty)"""
    body = _body({"name": name}, None, (), keep=())
    return await _request("POST", "/smallgroups/clusters", None, body)

@tool("smallgroups", "POST")
async def smallgroups_create_group_note(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the group to add the note to. Find it with smallgroups_list_groups (use `q` to search).")],
    content: Annotated[str, Field(description="Content (content) - REQUIRED. The content of the Note. (max 65535 characters; cannot be empty)")],
    pinned: Annotated[bool | None, Field(description="Pinned (pinned) - Optional. Whether the Note should be pinned.")] = None,
    visible_to: Annotated[dict[str, Any] | None, Field(description="Visible to (visible_to) - Optional. Object keys: ONE of these shapes: {just_me (one of true): The resource should only be visible to the current user} OR {user_group_ids (list of integer): List of user group IDs the resource is visible to. You are only able to add user group IDs for user groups you have access to -> ChurchSuite user-group IDs. No list tool; copy them from an existing record's visible_to, or ask a ChurchSuite admin.}")] = None,
) -> Any:
    """Create a group note. Create a Note against a Group in the Small Groups module.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the group to add the note to. Find it with smallgroups_list_groups (use `q` to search).
      - content (required): The content of the Note (max 65535 characters; cannot be empty)
      - pinned (optional): Whether the Note should be pinned
      - visible_to (optional): Visible to"""
    body = _body({"content": content, "pinned": pinned, "visible_to": visible_to}, None, (), keep=())
    return await _request("POST", f"/smallgroups/groups/{id}/notes", None, body)

@tool("smallgroups", "POST")
async def smallgroups_create_member(
    group_id: Annotated[int, Field(description="Group ID (group_id) - REQUIRED. The group ID the Member is being added to. HOW TO GET IT: Run smallgroups_list_groups and use the group `id`.")],
    person: Annotated[dict[str, Any], Field(description="Person (person) - REQUIRED. The Members's person details. Object keys: {type (one of children_child/addressbook_contact, required): The person's type, indicating whether they are a Child or Contact -> 'addressbook_contact' for a Contact or 'children_child' for a Child.; id (integer, required): The Contact or Child's ID -> A Contact ID (addressbook_list_contacts) or Child ID (children_list_children), matching `type`.}")],
    joined_date: Annotated[str, Field(description="Joined date (joined_date) - REQUIRED. Date the Member joined the group. (date as YYYY-MM-DD)")],
    status: Annotated[Literal["active", "pending"], Field(description="Status (status) - REQUIRED. The status of the Member. (one of: active, pending)")],
    privacy: Annotated[dict[str, Any], Field(description="Privacy (privacy) - REQUIRED. The Member's privacy settings. Object keys: {use_defaults (boolean, required): Use the Member's default privacy settings. If this is false `telephone_visible`, `mobile_visible` and `email_visible` must be set; telephone_visible (boolean): Telephone is visible; mobile_visible (boolean): Mobile is visible; email_visible (boolean): Email is visible}")],
    role_ids: Annotated[list[int] | None, Field(description="Role IDs (role_ids) - Optional. List of role_ids for any roles being assigned to the Member. HOW TO GET IT: Run smallgroups_list_roles and use the role `id`s.")] = None,
    send_email_confirmation: Annotated[bool | None, Field(description="Send email confirmation (send_email_confirmation) - Optional. Send the sign-up confirmation email to the Member you are adding and a sign-up notification to any existing members with the appropriate role.")] = None,
    override_capacity: Annotated[bool | None, Field(description="Override capacity (override_capacity) - Optional. Override the capacity of the group if adding the Member will take the group over capacity.")] = None,
) -> Any:
    """Create members. Create a member with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - group_id (required): The group ID the Member is being added to -> Run smallgroups_list_groups and use the group `id`.
      - person (required): The Members's person details
      - joined_date (required): Date the Member joined the group (date as YYYY-MM-DD)
      - status (required): The status of the Member (one of: active, pending)
      - privacy (required): The Member's privacy settings
      - role_ids (optional): List of role_ids for any roles being assigned to the Member -> Run smallgroups_list_roles and use the role `id`s.
      - send_email_confirmation (optional): Send the sign-up confirmation email to the Member you are adding and a sign-up notification to any existing members with the appropriate role
      - override_capacity (optional): Override the capacity of the group if adding the Member will take the group over capacity"""
    body = _body({"group_id": group_id, "person": person, "role_ids": role_ids, "joined_date": joined_date, "status": status, "privacy": privacy, "send_email_confirmation": send_email_confirmation, "override_capacity": override_capacity}, None, (), keep=())
    return await _request("POST", "/smallgroups/members", None, body)

@tool("smallgroups", "POST")
async def smallgroups_create_role(
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Role's name. (max 30 characters; cannot be empty)")],
    notify_signups: Annotated[bool | None, Field(description="Notify signups (notify_signups) - Optional. Whether to receive an email for each Sign-Up.")] = None,
    flyer_visible: Annotated[bool | None, Field(description="Flyer visible (flyer_visible) - Optional. Whether Members with this Role have their Contact details appear on the Group flyers.")] = None,
    my_visible: Annotated[bool | None, Field(description="My visible (my_visible) - Optional. Whether the Role will show within the Group Members list of My ChurchSuite.")] = None,
    my_edit: Annotated[bool | None, Field(description="My edit (my_edit) - Optional. Whether Members with this Role can edit the Group details and Member list through My ChurchSuite.")] = None,
    my_attendance: Annotated[bool | None, Field(description="My attendance (my_attendance) - Optional. Whether Members with this Role can submit attendance data through My ChurchSuite.")] = None,
) -> Any:
    """Create a role. Create a role with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - name (required): The Role's name (max 30 characters; cannot be empty)
      - notify_signups (optional): Whether to receive an email for each Sign-Up
      - flyer_visible (optional): Whether Members with this Role have their Contact details appear on the Group flyers
      - my_visible (optional): Whether the Role will show within the Group Members list of My ChurchSuite
      - my_edit (optional): Whether Members with this Role can edit the Group details and Member list through My ChurchSuite
      - my_attendance (optional): Whether Members with this Role can submit attendance data through My ChurchSuite"""
    body = _body({"name": name, "notify_signups": notify_signups, "flyer_visible": flyer_visible, "my_visible": my_visible, "my_edit": my_edit, "my_attendance": my_attendance}, None, (), keep=())
    return await _request("POST", "/smallgroups/roles", None, body)

@tool("smallgroups", "PUT")
async def smallgroups_edit_cluster(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the cluster to change. Find it with smallgroups_list_clusters (use `q` to search) or the matching get tool.")],
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Cluster's name. (max 50 characters; cannot be empty)")],
) -> Any:
    """Update a cluster. Update a single cluster identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the cluster to change. Find it with smallgroups_list_clusters (use `q` to search) or the matching get tool.
      - name (required): The Cluster's name (max 50 characters; cannot be empty)"""
    body = _body({"name": name}, None, (), keep=())
    return await _request("PUT", f"/smallgroups/clusters/{id}", None, body)

@tool("smallgroups", "PUT")
async def smallgroups_edit_member(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the member to change. Find it with smallgroups_list_members (use `q` to search) or the matching get tool.")],
    role_ids: Annotated[list[int], Field(description="Role IDs (role_ids) - REQUIRED. List of role_ids for any Roles being assigned to the Member. HOW TO GET IT: Run smallgroups_list_roles and use the role `id`s.")],
    joined_date: Annotated[str, Field(description="Joined date (joined_date) - REQUIRED. Date the Member joined the group. (date as YYYY-MM-DD)")],
    status: Annotated[Literal["active", "pending"], Field(description="Status (status) - REQUIRED. The current status of the Member. (one of: active, pending)")],
    privacy: Annotated[dict[str, Any], Field(description="Privacy (privacy) - REQUIRED. The Member's privacy settings. Object keys: {use_defaults (boolean, required): Use the Member's default privacy settings. If this is false `telephone_visible`, `mobile_visible` and `email_visible` must be set; telephone_visible (boolean): Telephone is visible; mobile_visible (boolean): Mobile is visible; email_visible (boolean): Email is visible}")],
) -> Any:
    """Update a member. Update a single member identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the member to change. Find it with smallgroups_list_members (use `q` to search) or the matching get tool.
      - role_ids (required): List of role_ids for any Roles being assigned to the Member -> Run smallgroups_list_roles and use the role `id`s.
      - joined_date (required): Date the Member joined the group (date as YYYY-MM-DD)
      - status (required): The current status of the Member (one of: active, pending)
      - privacy (required): The Member's privacy settings"""
    body = _body({"role_ids": role_ids, "joined_date": joined_date, "status": status, "privacy": privacy}, None, (), keep=())
    return await _request("PUT", f"/smallgroups/members/{id}", None, body)

@tool("smallgroups", "PUT")
async def smallgroups_edit_role(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the role to change. Find it with smallgroups_list_roles (use `q` to search) or the matching get tool.")],
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Role's name. (max 30 characters; cannot be empty)")],
    notify_signups: Annotated[bool, Field(description="Notify signups (notify_signups) - REQUIRED. Whether to receive an email for each Sign-Up.")],
    flyer_visible: Annotated[bool, Field(description="Flyer visible (flyer_visible) - REQUIRED. Whether Members with this Role have their Contact details appear on the Group flyers.")],
    my_visible: Annotated[bool, Field(description="My visible (my_visible) - REQUIRED. Whether the Role will show within the Group Members list of My ChurchSuite.")],
    my_edit: Annotated[bool, Field(description="My edit (my_edit) - REQUIRED. Whether Members with this Role can edit the Group details and Member list through My ChurchSuite.")],
    my_attendance: Annotated[bool, Field(description="My attendance (my_attendance) - REQUIRED. Whether Members with this Role can submit attendance data through My ChurchSuite.")],
) -> Any:
    """Update a role. Update a single role identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the role to change. Find it with smallgroups_list_roles (use `q` to search) or the matching get tool.
      - name (required): The Role's name (max 30 characters; cannot be empty)
      - notify_signups (required): Whether to receive an email for each Sign-Up
      - flyer_visible (required): Whether Members with this Role have their Contact details appear on the Group flyers
      - my_visible (required): Whether the Role will show within the Group Members list of My ChurchSuite
      - my_edit (required): Whether Members with this Role can edit the Group details and Member list through My ChurchSuite
      - my_attendance (required): Whether Members with this Role can submit attendance data through My ChurchSuite"""
    body = _body({"name": name, "notify_signups": notify_signups, "flyer_visible": flyer_visible, "my_visible": my_visible, "my_edit": my_edit, "my_attendance": my_attendance}, None, (), keep=())
    return await _request("PUT", f"/smallgroups/roles/{id}", None, body)

@tool("smallgroups", "DELETE")
async def smallgroups_delete_cluster(id: int) -> Any:
    """DESTRUCTIVE. Delete a cluster. Delete a single cluster identified by ID. Groups that are linked to this cluster will be unlinked. Smart Tags that reference this cluster will be affected."""
    return await _request("DELETE", f"/smallgroups/clusters/{id}", None)

@tool("smallgroups", "DELETE")
async def smallgroups_delete_member(id: int) -> Any:
    """DESTRUCTIVE. Delete a member. Delete a single member identified by ID."""
    return await _request("DELETE", f"/smallgroups/members/{id}", None)

@tool("smallgroups", "POST")
async def smallgroups_create_label(
    name: Annotated[str, Field(description="Name (name) - REQUIRED. The Label's name. (max 50 characters; cannot be empty)")],
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], Field(description="Colour (colour) - REQUIRED. The Label's colour. (one of: blue, red, orange, green, purple, pink, brown, grey)")],
    options: Annotated[list[str], Field(description="Options (options) - REQUIRED. List of options that belong to the Label.")],
    multiple_responses: Annotated[bool, Field(description="Multiple responses (multiple_responses) - REQUIRED. Whether the Label can have multiple responses.")],
    required: Annotated[bool, Field(description="Required (required) - REQUIRED. Whether the Label is required.")],
) -> Any:
    """Create a label. Create a label with the provided settings.
    
    WHAT YOU NEED TO PROVIDE
      - name (required): The Label's name (max 50 characters; cannot be empty)
      - colour (required): The Label's colour (one of: blue, red, orange, green, purple, pink, brown, grey)
      - options (required): List of options that belong to the Label
      - multiple_responses (required): Whether the Label can have multiple responses
      - required (required): Whether the Label is required"""
    body = _body({"name": name, "colour": colour, "options": options, "multiple_responses": multiple_responses, "required": required}, None, (), keep=())
    return await _request("POST", "/smallgroups/labels", None, body)

@tool("smallgroups", "PUT")
async def smallgroups_edit_label(
    id: Annotated[int, Field(description="ID (id) - REQUIRED. ID of the label to change. Find it with smallgroups_list_labels (use `q` to search) or the matching get tool.")],
    name: Annotated[str | None, Field(description="Name (name) - Optional - leave out to keep the current value. The Label's name. (max 50 characters; cannot be empty)")] = None,
    colour: Annotated[Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"] | None, Field(description="Colour (colour) - Optional - leave out to keep the current value. The Label's colour. (one of: blue, red, orange, green, purple, pink, brown, grey)")] = None,
    multiple_responses: Annotated[bool | None, Field(description="Multiple responses (multiple_responses) - Optional - leave out to keep the current value. Whether the Label can have multiple responses.")] = None,
    required: Annotated[bool | None, Field(description="Required (required) - Optional - leave out to keep the current value. Whether the Label is required.")] = None,
) -> Any:
    """Update a label. Update a single label identified by ID.
    
    WHAT YOU NEED TO PROVIDE
      - id (required): ID of the label to change. Find it with smallgroups_list_labels (use `q` to search) or the matching get tool.
      - name (optional): The Label's name (max 50 characters; cannot be empty)
      - colour (optional): The Label's colour (one of: blue, red, orange, green, purple, pink, brown, grey)
      - multiple_responses (optional): Whether the Label can have multiple responses
      - required (optional): Whether the Label is required"""
    body = _body({"name": name, "colour": colour, "multiple_responses": multiple_responses, "required": required}, None, (), keep=())
    return await _request("PUT", f"/smallgroups/labels/{id}", None, body)

@tool("smallgroups", "DELETE")
async def smallgroups_delete_label(id: int) -> Any:
    """DESTRUCTIVE. Delete a label. Delete a single label identified by ID. Label responses will be deleted from all groups. Module settings that reference this label will be affected."""
    return await _request("DELETE", f"/smallgroups/labels/{id}", None)

@tool("smallgroups", "DELETE")
async def smallgroups_delete_role(id: int) -> Any:
    """DESTRUCTIVE. Delete a role. Delete a single role identified by ID. Deleting a role is irreversible and will un-assign the role from every role holder for all past, active and future groups. Any Smart Tags that reference this role will be affected."""
    return await _request("DELETE", f"/smallgroups/roles/{id}", None)


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
