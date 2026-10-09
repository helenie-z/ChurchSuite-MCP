"""
MCP server for ChurchSuite (Core API v2, spec 2.109.0).

Modules (8): attendance, addressbook, calendar, children, network, planning, rotas, smallgroups.
Every GET, POST, PUT and DELETE endpoint in those modules is exposed as a tool, named
<module>_<list|get|create|edit|delete>_<resource>. This server CAN CHANGE AND DELETE DATA.

Run:  python server.py        (serves streamable HTTP at http://HOST:PORT/mcp)

Each MCP client sends its own ChurchSuite API credentials in the standard HTTP
Authorization header on every request, in either form:

      Authorization: Basic base64(<client ID>:<client secret>)
      Authorization: Bearer <client ID>:<client secret>

Server config (env, none of it secret):
      CHURCHSUITE_MODULES        comma-separated subset to enable (default: all eight)
      CHURCHSUITE_READ_ONLY      "true" hides every POST/PUT/DELETE tool and requests only
                                 the .read scopes (default: false)
      CHURCHSUITE_SCOPE          override the requested scopes
      CHURCHSUITE_API_BASE       (default https://api.churchsuite.com/v2)
      CHURCHSUITE_TOKEN_URL      (default https://login.churchsuite.com/oauth2/token)
      HOST / PORT                bind address for HTTP mode (default 0.0.0.0 / 8000)

Always serve this over HTTPS: the secret travels in the Authorization header.
"""

import base64
import hashlib
import os
import time
from typing import Any, Literal

import httpx
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

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


def _default_scope() -> str:
    parts = []
    for m in ENABLED:
        parts.append(f"{m}.read")
        if not READ_ONLY and m != "attendance":  # attendance has no write endpoints
            parts.append(f"{m}.write")
    return " ".join(parts)


SCOPE = os.environ.get("CHURCHSUITE_SCOPE") or _default_scope()
ALLOWED_PREFIXES = tuple(f"/{m}/" for m in ENABLED)

mcp = FastMCP(
    "churchsuite",
    instructions=(
        f"ChurchSuite access. Enabled modules: {', '.join(ENABLED)}. "
        + ("READ-ONLY mode: no create/edit/delete tools exist. " if READ_ONLY else
           "Create/edit/delete tools change real data: confirm with the user before using them, "
           "and never delete or overwrite anything the user did not explicitly ask for. ")
        + "List tools are paginated: pass page / per_page (1-250, default 50) and check "
        "pagination.next_page to continue. Array filters match ANY of the given values. "
        "Edit tools send only the fields you pass; to clear a nullable field, name it in "
        "set_null. Attendance: a Gathering has Metricsets containing Metrics; a Record is one "
        "Gathering on one date with Record Metrics (values) and, if people tracking is on, "
        "Record Contacts. Currency values are in the smallest unit (pence/cents). The children "
        "module holds children's personal data: handle it with care."
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


def _credentials() -> tuple[str, str]:
    """Read the caller's ChurchSuite credentials from the standard Authorization header.

    Accepts `Basic base64(id:secret)` or `Bearer id:secret`.
    """
    headers = get_http_headers(include_all=True)
    value = (headers.get("authorization") or "").strip()
    scheme, _, payload = value.partition(" ")
    payload = payload.strip()
    pair = ""
    if scheme.lower() == "basic" and payload:
        try:
            pair = base64.b64decode(payload).decode()
        except Exception:
            pair = ""
    elif scheme.lower() == "bearer":
        pair = payload
    cid, sep, secret = pair.partition(":")
    cid, secret = cid.strip(), secret.strip()
    if not sep or not cid or not secret:
        raise RuntimeError(
            "Missing or malformed credentials. Send an Authorization header of the form "
            "'Basic base64(client_id:client_secret)' or 'Bearer client_id:client_secret' "
            "with every request."
        )
    return cid, secret


# One cached token per credential pair, keyed by a hash so the secret is never used as a dict key.
_tokens: dict[str, tuple[str, float]] = {}


async def _auth_header(client: httpx.AsyncClient) -> dict[str, str]:
    """OAuth2 client-credentials token for the calling user, cached in memory and refreshed early."""
    cid, secret = _credentials()
    key = hashlib.sha256(f"{cid}\0{secret}".encode()).hexdigest()
    now = time.time()
    for k in [k for k, (_, exp) in _tokens.items() if exp <= now]:  # drop expired entries
        del _tokens[k]
    cached = _tokens.get(key)
    if cached and now < cached[1] - 60:
        token = cached[0]
    else:
        r = await client.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials", "scope": SCOPE},
            auth=(cid, secret),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Token request failed {r.status_code}: {r.text[:300]}")
        body = r.json()
        token = body["access_token"]
        _tokens[key] = (token, now + int(body.get("expires_in", 3600)))
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


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
        headers = await _auth_header(client)
        r = await client.request(
            method, f"{API_BASE}{path}", params=clean, headers=headers,
            **({"json": body} if body is not None else {}),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"ChurchSuite API {r.status_code}: {r.text[:500]}")
        if r.status_code == 204 or not r.text.strip():
            return {"status": r.status_code, "ok": True}
        return r.json()


def _body(
    values: dict[str, Any],
    set_null: list[str] | None,
    nullable: tuple[str, ...],
    keep: tuple[str, ...] = (),
) -> dict[str, Any]:
    # `keep` = required fields that may legitimately be null: always sent, even when None.
    body = {k: v for k, v in values.items() if v is not None or k in keep}
    for k in set_null or []:
        if k not in nullable:
            raise ValueError(f"'{k}' cannot be set to null. Nullable fields: {', '.join(nullable) or 'none'}")
        body[k] = None
    return body


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
    q: Filter records with a fuzzy string match against the following properties: first_name, last_name, formal_name, email, telephone, mobile, address, job…
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
async def addressbook_list_key_date_resources(page: int | None = None, per_page: int | None = None, key_date_ids: list[int] | None = None, contact_ids: list[int] | None = None) -> Any:
    """List key date resources. Returns a list of key date resources, optionally filtered by key date or contact ID. A key date resource records an instance of a key date for a contact on a given date.
    key_date_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    contact_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/addressbook/key_date_resources", {"page": page, "per_page": per_page, "key_date_ids[]": key_date_ids, "contact_ids[]": contact_ids})

@tool("addressbook", "GET")
async def addressbook_list_key_dates(page: int | None = None, per_page: int | None = None, q: str | None = None) -> Any:
    """List key dates. Returns a list of key dates, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/addressbook/key_dates", {"page": page, "per_page": per_page, "q": q})

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
async def addressbook_list_tags(page: int | None = None, per_page: int | None = None, q: str | None = None) -> Any:
    """List tags. Returns a list of tags, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/addressbook/tags", {"page": page, "per_page": per_page, "q": q})

@tool("addressbook", "POST")
async def addressbook_create_contact(first_name: str, last_name: str, all_sites: bool, site_ids: list[int], middle_name: str | None = None, title: str | None = None, formal_name: str | None = None, former_name: str | None = None, email: str | None = None, mobile: str | None = None, telephone: str | None = None, address: dict[str, Any] | None = None, sex: Literal["female", "male"] | None = None, date_of_birth: str | None = None, marital_status: Literal["single", "married", "engaged", "partnered", "separated", "divorced", "widowed"] | None = None, spouse_id: int | None = None, job: str | None = None, employer: str | None = None, work_telephone: str | None = None, custom_fields: list[Any] | None = None, communication: dict[str, Any] | None = None, login: dict[str, Any] | None = None, privacy: dict[str, Any] | None = None) -> Any:
    """Create a contact. Create a contact with the provided settings.
    first_name (required): The first name of the Contact.
    last_name (required): The last name of the Contact.
    all_sites (required): Whether the Contact belongs to All Sites.
    site_ids (required): An array of Site IDs that the Contact belongs to. If `all_sites` is `true`, this will be overridden. You are only able to add site IDs for sites you…
    middle_name: The middle name of the Contact.
    title: The title of the Contact.
    formal_name: The formal name of the Contact.
    former_name: Use this if a Contact has changed their last name and you want to record their previous last name. I.e. someone's maiden name.
    email: The Contact's email address.
    mobile: The Contact's mobile number.
    telephone: The Contact's telephone number.
    address: A street Address. Keys: line_1, line_2, line_3, city, county, country, postcode.
    sex: The sex of the Contact.
    date_of_birth: The Contact's date of birth.
    marital_status: The Contact's marital status.
    spouse_id: The Contact ID for the Contact's spouse.
    job: The job of the Contact.
    employer: The employer of the Contact.
    work_telephone: The Contact's work telephone number.
    custom_fields: Custom Field responses for this resource
    communication: The person's communication preferences. If no data is provided, the module default communication preferences will be used. Keys: general_email, gener…
    login: My ChurchSuite/CharitySuite login information. Keys: enabled, invited.
    privacy: Add the Contact's privacy settings. If no data is provided, the default privacy settings will be used. Keys: name_visible, address_visible, mobile_vi…"""
    return await _request("POST", "/addressbook/contacts", None, _body({"first_name": first_name, "last_name": last_name, "all_sites": all_sites, "site_ids": site_ids, "middle_name": middle_name, "title": title, "formal_name": formal_name, "former_name": former_name, "email": email, "mobile": mobile, "telephone": telephone, "address": address, "sex": sex, "date_of_birth": date_of_birth, "marital_status": marital_status, "spouse_id": spouse_id, "job": job, "employer": employer, "work_telephone": work_telephone, "custom_fields": custom_fields, "communication": communication, "login": login, "privacy": privacy}, None, ()))

@tool("addressbook", "POST")
async def addressbook_create_contact_note(id: int, content: str, pinned: bool | None = None, visible_to: Any | None = None) -> Any:
    """Create a contact note. Create a Note against a Contact in the Address Book module.
    content (required): The content of the Note.
    pinned: Whether the Note should be pinned.
    visible_to: """
    return await _request("POST", f"/addressbook/contacts/{id}/notes", None, _body({"content": content, "pinned": pinned, "visible_to": visible_to}, None, ()))

@tool("addressbook", "POST")
async def addressbook_create_flow_stage(flow_id: int, name: str, due_interval: int, due_period: Literal["days", "weeks", "months"], due_from: Literal["first_added", "previous_stage"], instructions: str | None = None, assigned_user_id: int | None = None, automate_time: str | None = None) -> Any:
    """Create a flow stage. Create a stage for a flow.
    flow_id (required): The Flow ID the Stage belongs to.
    name (required): The Stage's name.
    due_interval (required): The number of days, weeks or months used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when…
    due_period (required): The due period used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed fr…
    due_from (required): The relative point used for tracking a person linked to the Stage, either when they are first added to the flow or when they are processed from a pre…
    instructions: The instructions for the Stage.
    assigned_user_id: The ID of the user automatically assigned to people in this stage, or null if no user is assigned.
    automate_time: The UTC time the Stage will automatically process any due or overdue people if it is automated, or null if it is not automated."""
    return await _request("POST", "/addressbook/flow_stages", None, _body({"flow_id": flow_id, "name": name, "due_interval": due_interval, "due_period": due_period, "due_from": due_from, "instructions": instructions, "assigned_user_id": assigned_user_id, "automate_time": automate_time}, None, ()))

@tool("addressbook", "POST")
async def addressbook_create_flow_tracking(flow_id: int, stage_id: int, person: dict[str, Any]) -> Any:
    """Create a flow tracking. Create a flow tracking, linked to a person's process through a flow, with the provided settings.
    flow_id (required): The Flow ID the Tracking is linked to.
    stage_id (required): The Stage ID the Tracking is linked to.
    person (required): The Tracking's Person details. Keys: type, id."""
    return await _request("POST", "/addressbook/flow_trackings", None, _body({"flow_id": flow_id, "stage_id": stage_id, "person": person}, None, ()))

@tool("addressbook", "POST")
async def addressbook_create_key_date(name: str, colour: Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], visible_to: Any | None = None) -> Any:
    """Create a key date. Create a key date with the provided settings.
    name (required): The Key Date's name.
    colour (required): The Key Date's colour.
    visible_to: """
    return await _request("POST", "/addressbook/key_dates", None, _body({"name": name, "colour": colour, "visible_to": visible_to}, None, ()))

@tool("addressbook", "POST")
async def addressbook_create_key_date_resource(key_date_id: int, resource: dict[str, Any], date: str, description: str | None = None) -> Any:
    """Create a key date resource. Create a key date resource adding a contact to a key date
    key_date_id (required): The Key Date's ID.
    resource (required): The resource to link to the Key Date. Keys: id.
    date (required): The date of the Key Date Resource.
    description: The description of this Key Date Resource."""
    return await _request("POST", "/addressbook/key_date_resources", None, _body({"key_date_id": key_date_id, "resource": resource, "date": date, "description": description}, None, ()))

@tool("addressbook", "POST")
async def addressbook_create_tag(name: str, is_smart: bool, category_id: int | None = None, description: str | None = None, colour: Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"] | None = None, visible_to: Any | None = None, pinned_icon: Literal["tick", "cross", "archive", "warning", "flag", "magic", "star", "profile", "user", "couple", "group", "pledge", "education", "job", "gift", "role", "duration", "smiley-happy", "smiley-neutral", "smiley-sad"] | None = None) -> Any:
    """Create a tag
    name (required): The Tag's name.
    is_smart (required): Whether the Tag is smart or fixed.
    category_id: The Tag's Category ID. This sets the default colour of the Tag to the Category's colour.
    description: The description of the Tag.
    colour: The name of the colour of the Tag. If not specified, the Category colour will be used where applicable, falling back to 'blue'.
    visible_to: 
    pinned_icon: Pin this Tag by specifying an icon. Pinned tags give you a quick way to see your most important tags at a glance; they're displayed at the top of the…"""
    return await _request("POST", "/addressbook/tags", None, _body({"name": name, "is_smart": is_smart, "category_id": category_id, "description": description, "colour": colour, "visible_to": visible_to, "pinned_icon": pinned_icon}, None, ()))

@tool("addressbook", "POST")
async def addressbook_create_tag_resource(person: dict[str, Any], tag_id: int) -> Any:
    """Create a tag resource. Create a tag resource linking a contact to a tag
    person (required): The Person to link to the Tag. Keys: type, id.
    tag_id (required): The Tag's ID."""
    return await _request("POST", "/addressbook/tag_resources", None, _body({"person": person, "tag_id": tag_id}, None, ()))

@tool("addressbook", "PUT")
async def addressbook_edit_contact(id: int, first_name: str | None = None, last_name: str | None = None, middle_name: str | None = None, title: str | None = None, formal_name: str | None = None, former_name: str | None = None, email: str | None = None, mobile: str | None = None, telephone: str | None = None, address: dict[str, Any] | None = None, sex: Literal["female", "male"] | None = None, date_of_birth: str | None = None, marital_status: Literal["single", "married", "engaged", "partnered", "separated", "divorced", "widowed"] | None = None, spouse_id: int | None = None, job: str | None = None, employer: str | None = None, work_telephone: str | None = None, custom_fields: list[Any] | None = None, communication: dict[str, Any] | None = None, login: dict[str, Any] | None = None, privacy: dict[str, Any] | None = None, all_sites: bool | None = None, site_ids: list[int] | None = None, set_null: list[str] | None = None) -> Any:
    """Update a contact. Update a single contact identified by ID.
    first_name: The first name of the Contact.
    last_name: The last name of the Contact.
    middle_name: The middle name of the Contact.
    title: The title of the Contact.
    formal_name: The formal name of the Contact.
    former_name: Use this if a Contact has changed their last name and you want to record their previous last name. I.e. someone's maiden name.
    email: The Contact's email address.
    mobile: The Contact's mobile number.
    telephone: The Contact's telephone number.
    address: Edit a street Address. Keys: line_1, line_2, line_3, city, county, country, postcode.
    sex: The sex of the Contact.
    date_of_birth: The Contact's date of birth.
    marital_status: The Contact's marital status.
    spouse_id: The Contact ID for the Contact's spouse.
    job: The job of the Contact.
    employer: The employer of the Contact.
    work_telephone: The Contact's work telephone number.
    custom_fields: Custom Field responses for this resource
    communication: The person's communication preferences. Keys: general_email, general_sms, phone, post, rota_email, rota_sms.
    login: My ChurchSuite/CharitySuite login information. Keys: enabled, invited.
    privacy: Edit the Contact's privacy settings Keys: name_visible, address_visible, mobile_visible, email_visible, telephone_visible.
    all_sites: Whether the Contact belongs to All Sites.
    site_ids: An array of Site IDs that the Contact belongs to. If `all_sites` is `true`, this will be overridden. You are only able to add site IDs for sites you…
    set_null: names of nullable fields to clear (address, sex, date_of_birth, marital_status, spouse_id)."""
    return await _request("PUT", f"/addressbook/contacts/{id}", None, _body({"first_name": first_name, "last_name": last_name, "middle_name": middle_name, "title": title, "formal_name": formal_name, "former_name": former_name, "email": email, "mobile": mobile, "telephone": telephone, "address": address, "sex": sex, "date_of_birth": date_of_birth, "marital_status": marital_status, "spouse_id": spouse_id, "job": job, "employer": employer, "work_telephone": work_telephone, "custom_fields": custom_fields, "communication": communication, "login": login, "privacy": privacy, "all_sites": all_sites, "site_ids": site_ids}, set_null, ("address", "sex", "date_of_birth", "marital_status", "spouse_id", )))

@tool("addressbook", "PUT")
async def addressbook_edit_flow_stage(id: int, name: str, instructions: str, due_interval: int, due_period: Literal["days", "weeks", "months"], due_from: Literal["first_added", "previous_stage"], assigned_user_id: int | None, automate_time: str | None) -> Any:
    """Update a flow stage. Update a single stage identified by ID.
    name (required): The Stage's name.
    instructions (required): The instructions for the Stage.
    due_interval (required): The number of days, weeks or months used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when…
    due_period (required): The due period used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed fr…
    due_from (required): The relative point used for tracking a person linked to the Stage, either when they are first added to the flow or when they are processed from a pre…
    assigned_user_id (required): The ID of the user automatically assigned to people in this stage, or null if no user is assigned.
    automate_time (required): The UTC time the Stage will automatically process any due or overdue people if it is automated, or null if it is not automated."""
    return await _request("PUT", f"/addressbook/flow_stages/{id}", None, _body({"name": name, "instructions": instructions, "due_interval": due_interval, "due_period": due_period, "due_from": due_from, "assigned_user_id": assigned_user_id, "automate_time": automate_time}, None, (), keep=("assigned_user_id", "automate_time", )))

@tool("addressbook", "PUT")
async def addressbook_edit_flow_tracking(id: int, due_date: str, assigned_user_id: int | None) -> Any:
    """Update a flow tracking. Update a single flow tracking, linked to a person's process through a flow, identified by ID.
    due_date (required): The Tracking's due date.
    assigned_user_id (required): The Tracking's assigned user ID, or null if not assigned to a user."""
    return await _request("PUT", f"/addressbook/flow_trackings/{id}", None, _body({"due_date": due_date, "assigned_user_id": assigned_user_id}, None, (), keep=("assigned_user_id", )))

@tool("addressbook", "PUT")
async def addressbook_edit_key_date(id: int, name: str, colour: Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], visible_to: Any | None) -> Any:
    """Update a key date. Update a single key date identified by ID.
    name (required): The Key Date's name.
    colour (required): The Key Date's colour.
    visible_to (required): Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is f…"""
    return await _request("PUT", f"/addressbook/key_dates/{id}", None, _body({"name": name, "colour": colour, "visible_to": visible_to}, None, (), keep=("visible_to", )))

@tool("addressbook", "PUT")
async def addressbook_edit_key_date_resource(id: int, date: str, description: str) -> Any:
    """Update a key date resource. Update a single key date resource identified by ID.
    date (required): The date of the Key Date Resource.
    description (required): The description of this Key Date Resource."""
    return await _request("PUT", f"/addressbook/key_date_resources/{id}", None, _body({"date": date, "description": description}, None, ()))

@tool("addressbook", "PUT")
async def addressbook_edit_tag(id: int, name: str, category_id: int | None, description: str, colour: Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], visible_to: Any | None, pinned_icon: Literal["tick", "cross", "archive", "warning", "flag", "magic", "star", "profile", "user", "couple", "group", "pledge", "education", "job", "gift", "role", "duration", "smiley-happy", "smiley-neutral", "smiley-sad"] | None) -> Any:
    """Update a tag
    name (required): The Tag's name.
    category_id (required): The Tag's Category ID.
    description (required): The description of the Tag.
    colour (required): The name of the colour of the Tag.
    visible_to (required): Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is f…
    pinned_icon (required): Pin this Tag by specifying an icon. This will be displayed within the application."""
    return await _request("PUT", f"/addressbook/tags/{id}", None, _body({"name": name, "category_id": category_id, "description": description, "colour": colour, "visible_to": visible_to, "pinned_icon": pinned_icon}, None, (), keep=("category_id", "visible_to", "pinned_icon", )))

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
async def calendar_list_events(page: int | None = None, per_page: int | None = None, site_ids: list[int] | None = None, category_ids: list[int] | None = None, q: str | None = None, sequence_id: int | None = None, starts_after: str | None = None, starts_before: str | None = None, status: Literal["cancelled", "confirmed", "pending"] | None = None) -> Any:
    """List events. Returns a list of events, sorted by starts_at date.
    site_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    category_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    q: Filter records with a fuzzy string match against the following properties: name, starts_at.
    sequence_id: Filter records by the provided value.
    starts_after: Only include records that start after this date.
    starts_before: Only include records that start before this date.
    status: Filter records by the provided status."""
    return await _request("GET", "/calendar/events", {"page": page, "per_page": per_page, "site_ids[]": site_ids, "category_ids[]": category_ids, "q": q, "sequence_id": sequence_id, "starts_after": starts_after, "starts_before": starts_before, "status": status})

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
async def calendar_create_category(name: str, color: str, brand_id: int) -> Any:
    """Create a category. Create a category with the provided settings.
    name (required): The Category's name.
    color (required): The Category's hexadecimal colour value.
    brand_id (required): The Brand ID linked to the Category."""
    return await _request("POST", "/calendar/categories", None, _body({"name": name, "color": color, "brand_id": brand_id}, None, ()))

@tool("calendar", "POST")
async def calendar_create_event() -> Any:
    """Create an event. Create an event with the provided settings."""
    return await _request("POST", "/calendar/events", None, _body({}, None, ()))

@tool("calendar", "POST")
async def calendar_create_event_note(id: int, content: str, pinned: bool | None = None, visible_to: Any | None = None) -> Any:
    """Create an event note. Create a Note against an Event in the Calendar module.
    content (required): The content of the Note.
    pinned: Whether the Note should be pinned.
    visible_to: """
    return await _request("POST", f"/calendar/events/{id}/notes", None, _body({"content": content, "pinned": pinned, "visible_to": visible_to}, None, ()))

@tool("calendar", "POST")
async def calendar_create_signup() -> Any:
    """Create a sign-up. Create a sign-up with the provided settings."""
    return await _request("POST", "/calendar/signups", None, _body({}, None, ()))

@tool("calendar", "PUT")
async def calendar_edit_category(id: int, name: str, color: str, brand_id: int) -> Any:
    """Update a category. Update a single category identified by ID.
    name (required): The Category's name.
    color (required): The Category's hexadecimal colour value.
    brand_id (required): The Brand ID linked to the Category."""
    return await _request("PUT", f"/calendar/categories/{id}", None, _body({"name": name, "color": color, "brand_id": brand_id}, None, ()))

@tool("calendar", "PUT")
async def calendar_edit_event(id: int) -> Any:
    """Update an event. Update a single event identified by ID."""
    return await _request("PUT", f"/calendar/events/{id}", None, _body({}, None, ()))

@tool("calendar", "PUT")
async def calendar_edit_signup(id: int) -> Any:
    """Update a sign-up. Update a single sign-up identified by ID."""
    return await _request("PUT", f"/calendar/signups/{id}", None, _body({}, None, ()))

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
async def children_list_key_date_resources(page: int | None = None, per_page: int | None = None, key_date_ids: list[int] | None = None, child_ids: list[int] | None = None) -> Any:
    """List key date resources. Returns a list of key date resources, optionally filtered by key date or child ID. A key date resource records an instance of a key date for a child on a given date.
    key_date_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    child_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/children/key_date_resources", {"page": page, "per_page": per_page, "key_date_ids[]": key_date_ids, "child_ids[]": child_ids})

@tool("children", "GET")
async def children_list_key_dates(page: int | None = None, per_page: int | None = None, q: str | None = None) -> Any:
    """List key dates. Returns a list of key dates, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/children/key_dates", {"page": page, "per_page": per_page, "q": q})

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
async def children_list_tags(page: int | None = None, per_page: int | None = None, q: str | None = None) -> Any:
    """List tags. Returns a list of tags, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/children/tags", {"page": page, "per_page": per_page, "q": q})

@tool("children", "POST")
async def children_create_child(first_name: str, last_name: str, all_sites: bool, site_ids: list[int], middle_name: str | None = None, formal_name: str | None = None, sex: Literal["female", "male"] | None = None, date_of_birth: str | None = None, mobile: str | None = None, email: str | None = None, address: dict[str, Any] | None = None, custom_fields: list[Any] | None = None, communication: dict[str, Any] | None = None, school: str | None = None, school_year_offset: int | None = None, medical_short: str | None = None, medical: str | None = None, doctor_details: str | None = None, additional_needs: str | None = None, additional_info: str | None = None, login: dict[str, Any] | None = None, privacy: dict[str, Any] | None = None) -> Any:
    """Create a child. Create a child with the provided settings.
    first_name (required): The first name of the Child.
    last_name (required): The last name of the Child.
    all_sites (required): Whether the Child belongs to All Sites.
    site_ids (required): An array of Site IDs that the Child belongs to. If `all_sites` is `true`, this will be overridden. You are only able to add site IDs for sites you ha…
    middle_name: The middle name of the Child.
    formal_name: The formal name of the Child.
    sex: The sex of the Child.
    date_of_birth: The Child's date of birth.
    mobile: The Child's mobile number.
    email: The Child's email address.
    address: A street Address. Keys: line_1, line_2, line_3, city, county, country, postcode.
    custom_fields: Custom Field responses for this resource
    communication: The person's communication preferences. If no data is provided, the module default communication preferences will be used. Keys: general_email, gener…
    school: The Child's school.
    school_year_offset: Adjusted school year value based on the Child's age.
    medical_short: Condensed medical information about the Child.
    medical: Full medical information about the Child.
    doctor_details: Child's doctors details.
    additional_needs: Any additional needs declared about the Child.
    additional_info: Additional information about the Child.
    login: My ChurchSuite/CharitySuite login information. Keys: enabled, invited.
    privacy: The Child's privacy settings. If no data is provided, the default privacy settings will be used. Keys: name_visible, address_visible, mobile_visible,…"""
    return await _request("POST", "/children/children", None, _body({"first_name": first_name, "last_name": last_name, "all_sites": all_sites, "site_ids": site_ids, "middle_name": middle_name, "formal_name": formal_name, "sex": sex, "date_of_birth": date_of_birth, "mobile": mobile, "email": email, "address": address, "custom_fields": custom_fields, "communication": communication, "school": school, "school_year_offset": school_year_offset, "medical_short": medical_short, "medical": medical, "doctor_details": doctor_details, "additional_needs": additional_needs, "additional_info": additional_info, "login": login, "privacy": privacy}, None, ()))

@tool("children", "POST")
async def children_create_child_note(id: int, content: str, pinned: bool | None = None, visible_to: Any | None = None) -> Any:
    """Create a child note. Create a Note against a Child in the Children module.
    content (required): The content of the Note.
    pinned: Whether the Note should be pinned.
    visible_to: """
    return await _request("POST", f"/children/children/{id}/notes", None, _body({"content": content, "pinned": pinned, "visible_to": visible_to}, None, ()))

@tool("children", "POST")
async def children_create_flow_stage(flow_id: int, name: str, due_interval: int, due_period: Literal["days", "weeks", "months"], due_from: Literal["first_added", "previous_stage"], instructions: str | None = None, assigned_user_id: int | None = None, automate_time: str | None = None) -> Any:
    """Create a flow stage. Create a stage for a flow.
    flow_id (required): The Flow ID the Stage belongs to.
    name (required): The Stage's name.
    due_interval (required): The number of days, weeks or months used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when…
    due_period (required): The due period used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed fr…
    due_from (required): The relative point used for tracking a person linked to the Stage, either when they are first added to the flow or when they are processed from a pre…
    instructions: The instructions for the Stage.
    assigned_user_id: The ID of the user automatically assigned to people in this stage, or null if no user is assigned.
    automate_time: The UTC time the Stage will automatically process any due or overdue people if it is automated, or null if it is not automated."""
    return await _request("POST", "/children/flow_stages", None, _body({"flow_id": flow_id, "name": name, "due_interval": due_interval, "due_period": due_period, "due_from": due_from, "instructions": instructions, "assigned_user_id": assigned_user_id, "automate_time": automate_time}, None, ()))

@tool("children", "POST")
async def children_create_flow_tracking(flow_id: int, stage_id: int, person: dict[str, Any]) -> Any:
    """Create a flow tracking. Create a flow tracking, linked to a person's process through a flow, with the provided settings.
    flow_id (required): The Flow ID the Tracking is linked to.
    stage_id (required): The Stage ID the Tracking is linked to.
    person (required): The Tracking's Person details. Keys: type, id."""
    return await _request("POST", "/children/flow_trackings", None, _body({"flow_id": flow_id, "stage_id": stage_id, "person": person}, None, ()))

@tool("children", "POST")
async def children_create_key_date(name: str, colour: Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], visible_to: Any | None = None) -> Any:
    """Create a key date. Create a key date with the provided settings.
    name (required): The Key Date's name.
    colour (required): The Key Date's colour.
    visible_to: """
    return await _request("POST", "/children/key_dates", None, _body({"name": name, "colour": colour, "visible_to": visible_to}, None, ()))

@tool("children", "POST")
async def children_create_key_date_resource(key_date_id: int, resource: dict[str, Any], date: str, description: str | None = None) -> Any:
    """Create a key date resource. Create a key date resource adding a child to a key date
    key_date_id (required): The Key Date's ID.
    resource (required): The resource to link to the Key Date. Keys: id.
    date (required): The date of the Key Date Resource.
    description: The description of this Key Date Resource."""
    return await _request("POST", "/children/key_date_resources", None, _body({"key_date_id": key_date_id, "resource": resource, "date": date, "description": description}, None, ()))

@tool("children", "POST")
async def children_create_tag(name: str, is_smart: bool, category_id: int | None = None, description: str | None = None, colour: Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"] | None = None, visible_to: Any | None = None, pinned_icon: Literal["tick", "cross", "archive", "warning", "flag", "magic", "star", "profile", "user", "couple", "group", "pledge", "education", "job", "gift", "role", "duration", "smiley-happy", "smiley-neutral", "smiley-sad"] | None = None) -> Any:
    """Create a tag
    name (required): The Tag's name.
    is_smart (required): Whether the Tag is smart or fixed.
    category_id: The Tag's Category ID. This sets the default colour of the Tag to the Category's colour.
    description: The description of the Tag.
    colour: The name of the colour of the Tag. If not specified, the Category colour will be used where applicable, falling back to 'blue'.
    visible_to: 
    pinned_icon: Pin this Tag by specifying an icon. Pinned tags give you a quick way to see your most important tags at a glance; they're displayed at the top of the…"""
    return await _request("POST", "/children/tags", None, _body({"name": name, "is_smart": is_smart, "category_id": category_id, "description": description, "colour": colour, "visible_to": visible_to, "pinned_icon": pinned_icon}, None, ()))

@tool("children", "POST")
async def children_create_tag_resource(person: dict[str, Any], tag_id: int) -> Any:
    """Create a tag resource. Create a tag resource linking a child to a tag
    person (required): The Person to link to the Tag. Keys: type, id.
    tag_id (required): The Tag's ID."""
    return await _request("POST", "/children/tag_resources", None, _body({"person": person, "tag_id": tag_id}, None, ()))

@tool("children", "PUT")
async def children_edit_child(id: int, first_name: str | None = None, middle_name: str | None = None, last_name: str | None = None, formal_name: str | None = None, sex: Literal["female", "male"] | None = None, date_of_birth: str | None = None, mobile: str | None = None, email: str | None = None, address: dict[str, Any] | None = None, custom_fields: list[Any] | None = None, communication: dict[str, Any] | None = None, school: str | None = None, school_year_offset: int | None = None, medical_short: str | None = None, medical: str | None = None, doctor_details: str | None = None, additional_needs: str | None = None, additional_info: str | None = None, login: dict[str, Any] | None = None, privacy: dict[str, Any] | None = None, all_sites: bool | None = None, site_ids: list[int] | None = None, set_null: list[str] | None = None) -> Any:
    """Update a child. Update a single child identified by ID.
    first_name: The first name of the Child.
    middle_name: The middle name of the Child.
    last_name: The last name of the Child.
    formal_name: The formal name of the Child.
    sex: The sex of the Child.
    date_of_birth: The Child's date of birth.
    mobile: The Child's mobile number.
    email: The Child's email address.
    address: Edit a street Address. Keys: line_1, line_2, line_3, city, county, country, postcode.
    custom_fields: Custom Field responses for this resource
    communication: The person's communication preferences. Keys: general_email, general_sms, phone, post, rota_email, rota_sms.
    school: The Child's school.
    school_year_offset: Adjusted school year value based on the Child's age.
    medical_short: Condensed medical information about the Child.
    medical: Full medical information about the Child.
    doctor_details: Child's doctors details.
    additional_needs: Any additional needs declared about the Child.
    additional_info: Additional information about the Child.
    login: My ChurchSuite/CharitySuite login information. Keys: enabled, invited.
    privacy: The Child's privacy settings Keys: name_visible, address_visible, mobile_visible, email_visible.
    all_sites: Whether the Child belongs to All Sites.
    site_ids: An array of Site IDs that the Child belongs to. If `all_sites` is `true`, this will be overridden. You are only able to add site IDs for sites you ha…
    set_null: names of nullable fields to clear (sex, date_of_birth, email, address)."""
    return await _request("PUT", f"/children/children/{id}", None, _body({"first_name": first_name, "middle_name": middle_name, "last_name": last_name, "formal_name": formal_name, "sex": sex, "date_of_birth": date_of_birth, "mobile": mobile, "email": email, "address": address, "custom_fields": custom_fields, "communication": communication, "school": school, "school_year_offset": school_year_offset, "medical_short": medical_short, "medical": medical, "doctor_details": doctor_details, "additional_needs": additional_needs, "additional_info": additional_info, "login": login, "privacy": privacy, "all_sites": all_sites, "site_ids": site_ids}, set_null, ("sex", "date_of_birth", "email", "address", )))

@tool("children", "PUT")
async def children_edit_flow_stage(id: int, name: str, instructions: str, due_interval: int, due_period: Literal["days", "weeks", "months"], due_from: Literal["first_added", "previous_stage"], assigned_user_id: int | None, automate_time: str | None) -> Any:
    """Update a flow stage. Update a single stage identified by ID.
    name (required): The Stage's name.
    instructions (required): The instructions for the Stage.
    due_interval (required): The number of days, weeks or months used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when…
    due_period (required): The due period used for tracking a person linked to the Stage, either relative to when they are first added to the flow or when they are processed fr…
    due_from (required): The relative point used for tracking a person linked to the Stage, either when they are first added to the flow or when they are processed from a pre…
    assigned_user_id (required): The ID of the user automatically assigned to people in this stage, or null if no user is assigned.
    automate_time (required): The UTC time the Stage will automatically process any due or overdue people if it is automated, or null if it is not automated."""
    return await _request("PUT", f"/children/flow_stages/{id}", None, _body({"name": name, "instructions": instructions, "due_interval": due_interval, "due_period": due_period, "due_from": due_from, "assigned_user_id": assigned_user_id, "automate_time": automate_time}, None, (), keep=("assigned_user_id", "automate_time", )))

@tool("children", "PUT")
async def children_edit_flow_tracking(id: int, due_date: str, assigned_user_id: int | None) -> Any:
    """Update a flow tracking. Update a single flow tracking, linked to a person's process through a flow, identified by ID.
    due_date (required): The Tracking's due date.
    assigned_user_id (required): The Tracking's assigned user ID, or null if not assigned to a user."""
    return await _request("PUT", f"/children/flow_trackings/{id}", None, _body({"due_date": due_date, "assigned_user_id": assigned_user_id}, None, (), keep=("assigned_user_id", )))

@tool("children", "PUT")
async def children_edit_key_date(id: int, name: str, colour: Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], visible_to: Any | None) -> Any:
    """Update a key date. Update a single key date identified by ID.
    name (required): The Key Date's name.
    colour (required): The Key Date's colour.
    visible_to (required): Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is f…"""
    return await _request("PUT", f"/children/key_dates/{id}", None, _body({"name": name, "colour": colour, "visible_to": visible_to}, None, (), keep=("visible_to", )))

@tool("children", "PUT")
async def children_edit_key_date_resource(id: int, date: str, description: str) -> Any:
    """Update a key date resource. Update a single key date resource identified by ID.
    date (required): The date of the Key Date Resource.
    description (required): The description of this Key Date Resource."""
    return await _request("PUT", f"/children/key_date_resources/{id}", None, _body({"date": date, "description": description}, None, ()))

@tool("children", "PUT")
async def children_edit_tag(id: int, name: str, category_id: int | None, description: str, colour: Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], visible_to: Any | None, pinned_icon: Literal["tick", "cross", "archive", "warning", "flag", "magic", "star", "profile", "user", "couple", "group", "pledge", "education", "job", "gift", "role", "duration", "smiley-happy", "smiley-neutral", "smiley-sad"] | None) -> Any:
    """Update a tag
    name (required): The Tag's name.
    category_id (required): The Tag's Category ID.
    description (required): The description of the Tag.
    colour (required): The name of the colour of the Tag.
    visible_to (required): Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is f…
    pinned_icon (required): Pin this Tag by specifying an icon. This will be displayed within the application."""
    return await _request("PUT", f"/children/tags/{id}", None, _body({"name": name, "category_id": category_id, "description": description, "colour": colour, "visible_to": visible_to, "pinned_icon": pinned_icon}, None, (), keep=("category_id", "visible_to", "pinned_icon", )))

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
async def network_list_key_date_resources(page: int | None = None, per_page: int | None = None, key_date_ids: list[int] | None = None, organisation_ids: list[int] | None = None) -> Any:
    """List key date resources. Returns a list of key date resources, optionally filtered by key date or organisation ID. A key date resource records an instance of a key date for an organisation on a given date.
    key_date_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    organisation_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/network/key_date_resources", {"page": page, "per_page": per_page, "key_date_ids[]": key_date_ids, "organisation_ids[]": organisation_ids})

@tool("network", "GET")
async def network_list_key_dates(page: int | None = None, per_page: int | None = None, q: str | None = None) -> Any:
    """List key dates. Returns a list of key dates, sorted alphabetically by name.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/network/key_dates", {"page": page, "per_page": per_page, "q": q})

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
async def network_list_organisations(page: int | None = None, per_page: int | None = None, site_ids: list[int] | None = None, q: str | None = None, status: Literal["active", "archived"] | None = None) -> Any:
    """List organisations. Returns a list of organisation, sorted alphabetically by name.
    site_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    q: Filter records with a fuzzy string match against the following properties: name, reference (if enabled), meeting address city.
    status: Filter records by the provided status."""
    return await _request("GET", "/network/organisations", {"page": page, "per_page": per_page, "site_ids[]": site_ids, "q": q, "status": status})

@tool("network", "GET")
async def network_list_roles(page: int | None = None, per_page: int | None = None, q: str | None = None) -> Any:
    """List roles. Returns a list of roles, sorted by their order property.
    q: Filter records with a fuzzy string match against the following properties: name."""
    return await _request("GET", "/network/roles", {"page": page, "per_page": per_page, "q": q})

@tool("network", "POST")
async def network_create_key_date(name: str, colour: Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], visible_to: Any | None = None) -> Any:
    """Create a key date. Create a key date with the provided settings.
    name (required): The Key Date's name.
    colour (required): The Key Date's colour.
    visible_to: """
    return await _request("POST", "/network/key_dates", None, _body({"name": name, "colour": colour, "visible_to": visible_to}, None, ()))

@tool("network", "POST")
async def network_create_key_date_resource(key_date_id: int, resource: dict[str, Any], date: str, description: str | None = None) -> Any:
    """Create a key date resource. Create a key date resource adding an organisation to a key date
    key_date_id (required): The Key Date's ID.
    resource (required): The resource to link to the Key Date. Keys: id.
    date (required): The date of the Key Date Resource.
    description: The description of this Key Date Resource."""
    return await _request("POST", "/network/key_date_resources", None, _body({"key_date_id": key_date_id, "resource": resource, "date": date, "description": description}, None, ()))

@tool("network", "POST")
async def network_create_organisation_note(id: int, content: str, pinned: bool | None = None, visible_to: Any | None = None) -> Any:
    """Create a organisation note. Create a Note against a Organisation in the Network module.
    content (required): The content of the Note.
    pinned: Whether the Note should be pinned.
    visible_to: """
    return await _request("POST", f"/network/organisations/{id}/notes", None, _body({"content": content, "pinned": pinned, "visible_to": visible_to}, None, ()))

@tool("network", "PUT")
async def network_edit_key_date(id: int, name: str, colour: Literal["blue", "red", "orange", "green", "purple", "pink", "brown", "grey"], visible_to: Any | None) -> Any:
    """Update a key date. Update a single key date identified by ID.
    name (required): The Key Date's name.
    colour (required): The Key Date's colour.
    visible_to (required): Who the resource is visible to. Null if there are no restrictions. If `just_me` is true, `user_group_ids` must be empty or not set. If `just_me` is f…"""
    return await _request("PUT", f"/network/key_dates/{id}", None, _body({"name": name, "colour": colour, "visible_to": visible_to}, None, (), keep=("visible_to", )))

@tool("network", "PUT")
async def network_edit_key_date_resource(id: int, date: str, description: str) -> Any:
    """Update a key date resource. Update a single key date resource identified by ID.
    date (required): The date of the Key Date Resource.
    description (required): The description of this Key Date Resource."""
    return await _request("PUT", f"/network/key_date_resources/{id}", None, _body({"date": date, "description": description}, None, ()))

@tool("network", "DELETE")
async def network_delete_key_date(id: int) -> Any:
    """DESTRUCTIVE. Delete a key date. Delete a single key date identified by ID."""
    return await _request("DELETE", f"/network/key_dates/{id}", None)

@tool("network", "DELETE")
async def network_delete_key_date_resource(id: int) -> Any:
    """DESTRUCTIVE. Delete a key date resource. Delete a single key date resource identified by an ID, removing an organisation from a key date."""
    return await _request("DELETE", f"/network/key_date_resources/{id}", None)



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
async def planning_list_plan_items(plan_ids: list[int], page: int | None = None, per_page: int | None = None) -> Any:
    """List plan items. Returns a list of plan items, sorted by ID.
    plan_ids (required): Filter records to those matching values in the provided array. This filter uses the logical OR operator."""
    return await _request("GET", "/planning/plan_items", {"plan_ids[]": plan_ids, "page": page, "per_page": per_page})

@tool("planning", "GET")
async def planning_list_plans(page: int | None = None, per_page: int | None = None, site_ids: list[int] | None = None, q: str | None = None, starts_after: str | None = None, starts_before: str | None = None, status: Literal["draft", "published"] | None = None) -> Any:
    """List plans. Returns a list of plans, sorted by date and time.
    site_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    q: Filter records with a fuzzy string match against the following properties: name.
    starts_after: Only include records that start after this date.
    starts_before: Only include records that start before this date.
    status: Filter records by the provided status."""
    return await _request("GET", "/planning/plans", {"page": page, "per_page": per_page, "site_ids[]": site_ids, "q": q, "starts_after": starts_after, "starts_before": starts_before, "status": status})

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
async def planning_create_plan_note(id: int, content: str, pinned: bool | None = None, visible_to: Any | None = None) -> Any:
    """Create a plan note. Create a Note against a Plan in the Planning module.
    content (required): The content of the Note.
    pinned: Whether the Note should be pinned.
    visible_to: """
    return await _request("POST", f"/planning/plans/{id}/notes", None, _body({"content": content, "pinned": pinned, "visible_to": visible_to}, None, ()))



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
async def rotas_list_unavailability(page: int | None = None, per_page: int | None = None, contact_ids: list[int] | None = None, child_ids: list[int] | None = None, ministry_ids: list[int] | None = None, starts_before: str | None = None, starts_after: str | None = None) -> Any:
    """List unavailability. Returns a list of unavailability, sorted by date created.
    contact_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    child_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    ministry_ids: Filter records to those matching values in the provided array. This filter uses the logical OR operator.
    starts_before: Only include records that start before this date.
    starts_after: Only include records that start after this date."""
    return await _request("GET", "/rotas/unavailability", {"page": page, "per_page": per_page, "contact_ids[]": contact_ids, "child_ids[]": child_ids, "ministry_ids[]": ministry_ids, "starts_before": starts_before, "starts_after": starts_after})

@tool("rotas", "POST")
async def rotas_create_unavailability(person: dict[str, Any], starts_at: str, ends_at: str, description: str | None = None) -> Any:
    """Create unavailability. Create unavailability, with the provided settings.
    person (required): The Person this Unavailability record belongs to. Keys: type, id.
    starts_at (required): The date and time the Unavailability starts. To create unavailability lasting all day, start at 00:00:00 and end at 23:59:59.
    ends_at (required): The date and time the Unavailability ends. To create unavailability lasting all day, start at 00:00:00 and end at 23:59:59.
    description: The Unavailability's description."""
    return await _request("POST", "/rotas/unavailability", None, _body({"person": person, "starts_at": starts_at, "ends_at": ends_at, "description": description}, None, ()))

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
async def smallgroups_create_cluster(name: str) -> Any:
    """Create a cluster. Create a cluster with the provided settings.
    name (required): The Cluster's name."""
    return await _request("POST", "/smallgroups/clusters", None, _body({"name": name}, None, ()))

@tool("smallgroups", "POST")
async def smallgroups_create_group_note(id: int, content: str, pinned: bool | None = None, visible_to: Any | None = None) -> Any:
    """Create a group note. Create a Note against a Group in the Small Groups module.
    content (required): The content of the Note.
    pinned: Whether the Note should be pinned.
    visible_to: """
    return await _request("POST", f"/smallgroups/groups/{id}/notes", None, _body({"content": content, "pinned": pinned, "visible_to": visible_to}, None, ()))

@tool("smallgroups", "POST")
async def smallgroups_create_member(group_id: int, person: dict[str, Any], joined_date: str, status: Literal["active", "pending"], privacy: dict[str, Any], role_ids: list[int] | None = None, send_email_confirmation: bool | None = None, override_capacity: bool | None = None) -> Any:
    """Create members. Create a member with the provided settings.
    group_id (required): The group ID the Member is being added to.
    person (required): The Members's person details. Keys: type, id.
    joined_date (required): Date the Member joined the group.
    status (required): The status of the Member.
    privacy (required): The Member's privacy settings. Keys: use_defaults, telephone_visible, mobile_visible, email_visible.
    role_ids: List of role_ids for any roles being assigned to the Member.
    send_email_confirmation: Send the sign-up confirmation email to the Member you are adding and a sign-up notification to any existing members with the appropriate role.
    override_capacity: Override the capacity of the group if adding the Member will take the group over capacity."""
    return await _request("POST", "/smallgroups/members", None, _body({"group_id": group_id, "person": person, "joined_date": joined_date, "status": status, "privacy": privacy, "role_ids": role_ids, "send_email_confirmation": send_email_confirmation, "override_capacity": override_capacity}, None, ()))

@tool("smallgroups", "POST")
async def smallgroups_create_role(name: str, notify_signups: bool | None = None, flyer_visible: bool | None = None, my_visible: bool | None = None, my_edit: bool | None = None, my_attendance: bool | None = None) -> Any:
    """Create a role. Create a role with the provided settings.
    name (required): The Role's name.
    notify_signups: Whether to receive an email for each Sign-Up.
    flyer_visible: Whether Members with this Role have their Contact details appear on the Group flyers.
    my_visible: Whether the Role will show within the Group Members list of My ChurchSuite.
    my_edit: Whether Members with this Role can edit the Group details and Member list through My ChurchSuite.
    my_attendance: Whether Members with this Role can submit attendance data through My ChurchSuite."""
    return await _request("POST", "/smallgroups/roles", None, _body({"name": name, "notify_signups": notify_signups, "flyer_visible": flyer_visible, "my_visible": my_visible, "my_edit": my_edit, "my_attendance": my_attendance}, None, ()))

@tool("smallgroups", "PUT")
async def smallgroups_edit_cluster(id: int, name: str) -> Any:
    """Update a cluster. Update a single cluster identified by ID.
    name (required): The Cluster's name."""
    return await _request("PUT", f"/smallgroups/clusters/{id}", None, _body({"name": name}, None, ()))

@tool("smallgroups", "PUT")
async def smallgroups_edit_member(id: int, role_ids: list[int], joined_date: str, status: Literal["active", "pending"], privacy: dict[str, Any]) -> Any:
    """Update a member. Update a single member identified by ID.
    role_ids (required): List of role_ids for any Roles being assigned to the Member.
    joined_date (required): Date the Member joined the group.
    status (required): The current status of the Member.
    privacy (required): The Member's privacy settings. Keys: use_defaults, telephone_visible, mobile_visible, email_visible."""
    return await _request("PUT", f"/smallgroups/members/{id}", None, _body({"role_ids": role_ids, "joined_date": joined_date, "status": status, "privacy": privacy}, None, ()))

@tool("smallgroups", "PUT")
async def smallgroups_edit_role(id: int, name: str, notify_signups: bool, flyer_visible: bool, my_visible: bool, my_edit: bool, my_attendance: bool) -> Any:
    """Update a role. Update a single role identified by ID.
    name (required): The Role's name.
    notify_signups (required): Whether to receive an email for each Sign-Up.
    flyer_visible (required): Whether Members with this Role have their Contact details appear on the Group flyers.
    my_visible (required): Whether the Role will show within the Group Members list of My ChurchSuite.
    my_edit (required): Whether Members with this Role can edit the Group details and Member list through My ChurchSuite.
    my_attendance (required): Whether Members with this Role can submit attendance data through My ChurchSuite."""
    return await _request("PUT", f"/smallgroups/roles/{id}", None, _body({"name": name, "notify_signups": notify_signups, "flyer_visible": flyer_visible, "my_visible": my_visible, "my_edit": my_edit, "my_attendance": my_attendance}, None, ()))

@tool("smallgroups", "DELETE")
async def smallgroups_delete_cluster(id: int) -> Any:
    """DESTRUCTIVE. Delete a cluster. Delete a single cluster identified by ID. Groups that are linked to this cluster will be unlinked. Smart Tags that reference this cluster will be affected."""
    return await _request("DELETE", f"/smallgroups/clusters/{id}", None)

@tool("smallgroups", "DELETE")
async def smallgroups_delete_member(id: int) -> Any:
    """DESTRUCTIVE. Delete a member. Delete a single member identified by ID."""
    return await _request("DELETE", f"/smallgroups/members/{id}", None)

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
