# Architecture Diagram

```mermaid
flowchart LR
    Client[Client / OpenAI-compatible caller]
    Uvicorn[Uvicorn server\nrun.sh foreground process]
    API[FastAPI app\napp.py]
    Auth[Bearer token validation\nOPENAI_API_KEY]
    Queue[GPU semaphore\nMAX_CONCURRENT]
    Pipeline[Diffusers AutoPipelineForText2Image]
    GPU[(CUDA / CPU runtime)]

    Client -->|POST /v1/images/generations| Uvicorn
    Client -->|POST /generate_image/| Uvicorn
    Uvicorn --> API
    API --> Auth
    API --> Queue
    Queue --> Pipeline
    Pipeline --> GPU
    GPU --> Pipeline
    Pipeline --> API
    API -->|PNG/JPEG or b64 JSON| Client
```

## Notes
- `run.sh` is idempotent: it recreates/uses `~/venv/<project-name>` and only reinstalls dependencies when `requirements.txt` changes.
- `run.sh` is systemd friendly because it runs in foreground and `exec`s uvicorn.
- `upgrade.sh` upgrades toolchain + project dependencies in the same virtualenv path.
