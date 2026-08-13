# api-txt2image

Minimal, token-protected, OpenAI-compatible text-to-image API for NVIDIA H100
and DGX Spark (aarch64), built with FastAPI, Diffusers, PyTorch, and `uv`.

## Start

```bash
cp .env.example .env               # change OPENAI_API_KEY
./install.sh                        # safe for first install and upgrades
source run.sh [IP] [PORT]           # PORT is optional; a free port is selected
```

The environment is always stored at `~/venv/<project-directory-name>`. Both
scripts are idempotent; normal starts only install when `requirements.txt`
changes. `./upgrade.sh` remains as an alias for `./install.sh`.

Generate an image using the widely supported OpenAI Images API shape:

```bash
BASE_URL=http://127.0.0.1:8000
curl "$BASE_URL/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a small robot painting","size":"1024x1024"}'
```

The response contains `data[].b64_json`. Health is available without a token at
`GET /healthz`. Interactive API documentation is at `/docs`.

## User service

`run.sh` stays in the foreground and uses `exec`, so it works directly with
`systemctl --user` (a shell `source` is not needed inside the unit):

```ini
[Unit]
Description=OpenAI-compatible text-to-image API
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/api-txt2image
EnvironmentFile=%h/api-txt2image/.env
ExecStart=%h/api-txt2image/run.sh 0.0.0.0 8000
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
mkdir -p ~/.config/systemd/user
# save the unit as ~/.config/systemd/user/api-txt2image.service
systemctl --user daemon-reload
systemctl --user enable --now api-txt2image
```

Important configuration and defaults are documented in `.env.example`.
