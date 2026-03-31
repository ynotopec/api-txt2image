# Architecture Diagram

```mermaid
flowchart LR
    Client[Client / OpenAI-compatible caller]
    Uvicorn[Uvicorn server\nrun.sh foreground process]
    API[FastAPI app\napp.py]
    Auth[Bearer token validation\nOPENAI_API_KEY]
    Queue[GPU semaphore\nMAX_CONCURRENT]
    Pipeline[Diffusers AutoPipelineForText2Image]
    Idle[Idle monitor task\nIDLE_UNLOAD_SECONDS]
    GPU[(CUDA / CPU runtime)]

    Client -->|POST /v1/images/generations| Uvicorn
    Client -->|GET /healthz| Uvicorn
    Uvicorn --> API
    API --> Auth
    API --> Queue
    Queue --> Pipeline
    API --> Idle
    Idle -->|idle timeout| Pipeline
    Pipeline --> GPU
    GPU --> Pipeline
    Pipeline --> API
    API -->|b64 JSON| Client
```

## Operational notes

- `run.sh` is idempotent: it reuses `~/venv/<project-name>` and reinstalls dependencies only when `requirements.txt` changes.
- `run.sh` runs in the foreground and uses `exec`, so it is compatible with systemd/container supervisors.
- `upgrade.sh` performs explicit dependency upgrades in the same venv path.
- `app.py` can unload the model from GPU when idle (`IDLE_UNLOAD_SECONDS`) and lazily reload on the next request.
