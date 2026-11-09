# Hermes Agent integration

[Hermes Agent](https://hermes-agent.nousresearch.com/) talks to the service over
MCP (Streamable HTTP). Two steps: register the MCP server, install the skill.

## 1. Register the MCP server

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  transcribation:
    url: "http://127.0.0.1:8000/mcp"        # or https://transcribe.example.com/mcp behind a proxy
    headers:
      Authorization: "Bearer ${TRANSCRIBE_API_TOKEN}"
    timeout: 300          # per tool call; transcribe_video waits up to wait_seconds (90 by default)
    connect_timeout: 30
```

Put the token in `~/.hermes/.env`:

```bash
TRANSCRIBE_API_TOKEN=<the same value as in the service's .env>
```

Then run `/reload-mcp` in a Hermes chat (or restart Hermes). The agent gets three
tools: `transcribe_video`, `get_transcript`, `list_transcripts`.

## 2. Install the skill (recommended)

The skill tells the agent when to use the tools and how to read the transcript
format (speakers, timestamps, low-confidence marks, paging).

```bash
mkdir -p ~/.hermes/skills/youtube-transcribe
cp integrations/hermes/SKILL.md ~/.hermes/skills/youtube-transcribe/SKILL.md
```

## Check

Ask Hermes: *"Summarize https://youtu.be/jNQXAC9IVRw"* — it should call
`transcribe_video` and answer from the transcript.

If Hermes runs on another machine, expose the service through a reverse proxy
with TLS (or Tailscale) and always set `TRANSCRIBE_API_TOKEN`.
