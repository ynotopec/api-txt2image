# api-txt2image

Minimal OpenAI-compatible text-to-image API powered by FastAPI + Diffusers.

## What this repo now optimizes for

- **Simple operations**: one app file (`app.py`) and two scripts (`run.sh`, `upgrade.sh`).
- **Idempotent startup**: `run.sh` only reinstalls dependencies when `requirements.txt` changes.
- **Predictable virtualenv**: always uses `~/venv/<project-name>`.
- **`uv`-based workflow**: fast env + package management.
- **Foreground runtime**: `run.sh` `exec`s uvicorn (systemd/container friendly).
- **Single mandatory setting**: only `OPENAI_API_KEY` is required.

## Quick start

```bash
cp .env.example .env
./upgrade.sh
./run.sh 0.0.0.0 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

## Configuration

Set values in `.env` (template provided in `.env.example`).

- **Required**: `OPENAI_API_KEY`
- **Optional**: all other variables have defaults in `app.py` (`MODEL_ID` defaults to `dataautogpt3/OpenDalle`)

## Scripts

### `./run.sh [IP] [PORT]`

- Creates/repairs `~/venv/<project-name>`.
- Installs deps if and only if `requirements.txt` changed.
- Loads `.env` if present.
- Fails fast if `OPENAI_API_KEY` is missing.
- Starts uvicorn in foreground (`exec ...`) for process managers.

Examples:

```bash
./run.sh
./run.sh 127.0.0.1 9000
```

### `./upgrade.sh`

- Ensures `uv` is installed.
- Ensures `~/venv/<project-name>` exists.
- Upgrades packaging tools + project requirements.

Run anytime:

```bash
./upgrade.sh
```

## systemd example

Use direct exec (no `bash -c`) and prefer `0.0.0.0` unless you must bind a specific interface IP.

```ini
[Unit]
Description=api-txt2image
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=ailab
WorkingDirectory=/home/ailab/api-txt2image
EnvironmentFile=/home/ailab/api-txt2image/.env
ExecStart=/home/ailab/api-txt2image/run.sh 0.0.0.0 8522
Restart=always
RestartSec=10
StandardOutput=append:/var/log/api-txt2image_monitor.log
StandardError=append:/var/log/api-txt2image_monitor.log

[Install]
WantedBy=multi-user.target
```

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the Mermaid diagram and flow notes.

## API

- `GET /healthz`
- `POST /v1/images/generations`

The image generation endpoint returns OpenAI-style `b64_json` payloads.
