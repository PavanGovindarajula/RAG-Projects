"""MCP gateway backed by an explicitly configured incident evidence HTTP API."""
import json
import os
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('incident-evidence-gateway')

@mcp.tool()
def lookup_incident_evidence(issue: str, version: str) -> str:
    """Search the configured incident evidence API for an issue and product version."""
    base = os.environ.get('INCIDENT_API_URL', '')
    parsed = urlparse(base)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError('INCIDENT_API_URL must be an HTTPS URL without query or fragment')
    headers = {'Accept': 'application/json'}
    if os.environ.get('INCIDENT_API_TOKEN'):
        headers['Authorization'] = 'Bearer ' + os.environ['INCIDENT_API_TOKEN']
    req = Request(base + '?' + urlencode({'issue': issue, 'version': version}), headers=headers)
    with urlopen(req, timeout=10) as response:
        body = response.read(1_000_001)
    if len(body) > 1_000_000:
        raise ValueError('API response too large')
    data = json.loads(body)
    if not isinstance(data, list) or len(data) > 20:
        raise ValueError('Expected a list of at most 20 evidence records')
    return json.dumps(data)

def main():
    mcp.run(transport='stdio')

if __name__ == '__main__':
    main()
