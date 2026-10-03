# ChurchSuite MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server for the
[ChurchSuite](https://churchsuite.com) Core API v2. It exposes **all GET, POST, PUT and DELETE
endpoint** in eight modules as MCP tools, so an AI assistant can read and manage your
ChurchSuite data.

> **Warning:** this server can create, change and delete real data. Use a ChurchSuite API client with only the permissions you need, or run it in
> read-only mode (see below).

## How authentication works

The server stores **no ChurchSuite credentials**. Each MCP client sends its own API credentials
as HTTP headers on every request:

| Header | Value |
| --- | --- |
| `X-ChurchSuite-Client-Id` | Your ChurchSuite API client ID |
| `X-ChurchSuite-Client-Secret` | Your ChurchSuite API client secret |

The server uses them to fetch an OAuth2 client-credentials token from ChurchSuite (cached in
memory per credential pair and refreshed before expiry) and calls the API on that user's behalf.
Several people can share one deployment, each with their own credentials.

**Always deploy behind HTTPS**, since the secret travels in a request header. Never log request
headers in your proxy or platform.

## Modules and tools

`attendance`, `addressbook`, `calendar`, `children`, `network`, `planning`, `rotas`, `smallgroups`

Tools are named `<module>_<list|get|create|edit|delete>_<resource>`, for example
`attendance_list_gatherings`, `calendar_create_event`, `smallgroups_delete_member`.

- List tools are paginated (`page`, `per_page` 1-250, default 50).
- Edit tools send only the fields you pass; to clear a nullable field, name it in `set_null`.
- Delete tools are annotated as destructive so compliant clients will ask for confirmation.

## Getting API credentials

1. In ChurchSuite, set up an API client following the
   [Developer API guide](https://support.churchsuite.com/article/454-developer-api).
2. Grant it the scopes for the modules you want (`<module>.read`, and `<module>.write` if you
   want changes).
3. Keep the client ID and secret somewhere safe. You will put them in your MCP client config.

## Configuration

All server settings are optional environment variables and none are secret
(see [`.env.example`](.env.example)):

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHURCHSUITE_MODULES` | all eight | Comma-separated subset of modules to enable |
| `CHURCHSUITE_READ_ONLY` | `false` | `true` removes every create/edit/delete tool and requests only `.read` scopes |
| `CHURCHSUITE_SCOPE` | derived | Override the OAuth scopes requested |
| `CHURCHSUITE_API_BASE` | `https://api.churchsuite.com/v2` | API base URL |
| `CHURCHSUITE_TOKEN_URL` | `https://login.churchsuite.com/oauth2/token` | OAuth token endpoint |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Port (most platforms set this for you) |

Tip: the requested scopes must be ones your API client has been granted. If you only enabled
read access in ChurchSuite, set `CHURCHSUITE_READ_ONLY=true`.

## Run locally

```bash
pip install -r requirements.txt
python server.py
```

The MCP endpoint is then `http://localhost:8000/mcp` (streamable HTTP). With
[uv](https://docs.astral.sh/uv/) you can instead run `uv run server.py`.

## Run with Docker

```bash
docker build -t churchsuite-mcp .
docker run --rm -p 8000:8000 churchsuite-mcp

# read-only, attendance + calendar only
docker run --rm -p 8000:8000 \
  -e CHURCHSUITE_READ_ONLY=true \
  -e CHURCHSUITE_MODULES=attendance,calendar \
  churchsuite-mcp
```

## Deploying

Any host that can run a Python process or a container will work, as long as it terminates HTTPS
and forwards `/mcp`.

- **Container platforms (Fly.io, Render, Railway, Cloud Run, etc.):** deploy the included
  `Dockerfile`. Make sure the platform's port matches `PORT` (the server reads it from the
  environment).
- **Prefect Horizon / FastMCP-style hosts:** point the platform at this repo with the entrypoint
  `server.py:mcp`.
- Set `CHURCHSUITE_READ_ONLY` / `CHURCHSUITE_MODULES` in the platform's environment settings if
  you want to restrict the deployment.

## Connecting a client

Your client must support **custom HTTP headers** on remote MCP servers.

**Claude Code**

```bash
claude mcp add --transport http churchsuite https://YOUR-HOST/mcp \
  --header "X-ChurchSuite-Client-Id: YOUR_CLIENT_ID" \
  --header "X-ChurchSuite-Client-Secret: YOUR_CLIENT_SECRET"
```

**Generic JSON config** (clients that accept `url` + `headers`)

```json
{
  "mcpServers": {
    "churchsuite": {
      "url": "https://YOUR-HOST/mcp",
      "headers": {
        "X-ChurchSuite-Client-Id": "YOUR_CLIENT_ID",
        "X-ChurchSuite-Client-Secret": "YOUR_CLIENT_SECRET"
      }
    }
  }
}
```

**Quick test with curl** (initialize handshake; a missing header surfaces as a tool error when a
tool is called)

```bash
curl -s https://YOUR-HOST/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "X-ChurchSuite-Client-Id: YOUR_CLIENT_ID" \
  -H "X-ChurchSuite-Client-Secret: YOUR_CLIENT_SECRET" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
```

## Security notes

- Credentials are read per request and never written to disk or logged by this server.
- Only paths under the enabled modules (`/attendance/`, `/calendar/`, ...) can be called.
- Anyone who can reach the URL **and** holds valid ChurchSuite credentials can act with those
  credentials. Consider putting the deployment behind an IP allow-list or an authenticating
  proxy as well.
- Rotate the ChurchSuite client secret if it is ever exposed.

## Project layout

```
server.py            MCP server (all tools)
requirements.txt     Python dependencies
pyproject.toml       Project metadata (for uv)
Dockerfile           Container image
.env.example         Non-secret server settings
.github/workflows/   CI (syntax check, import check, Docker build)
```

## License

MIT. See [LICENSE](LICENSE).
