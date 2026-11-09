# Security policy

Please report vulnerabilities privately via
[GitHub security advisories](https://github.com/fakktors/TranscribationServicesPy/security/advisories/new),
not in public issues. You should get a reply within a week.

## Deployment notes

The service downloads media from URLs that clients send, so treat it as a
network-facing component:

- Always set `TRANSCRIBE_API_TOKEN` when the port is reachable by anyone but you.
  `compose.yaml` publishes on `127.0.0.1` only by default.
- Keep `TRANSCRIBE_ALLOWED_HOSTS` narrow (YouTube by default). `*` lets clients
  make the server fetch arbitrary URLs through yt-dlp.
- Terminate TLS in a reverse proxy (Caddy, nginx, Traefik) or use a private
  network (Tailscale/WireGuard) — the token travels in a header.
- The container runs as an unprivileged user (uid 1000) and needs no capabilities.
- Never commit `.env` or `cookies.txt`.
