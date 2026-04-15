# /home/ailab/api-txt2image/app.py
import os
import io
import time
import base64
import asyncio
import secrets
import warnings
from typing import Optional, Literal, List, Tuple

import torch
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from diffusers import AutoPipelineForText2Image

warnings.filterwarnings(
    "ignore",
    message=r".*Siglip2ImageProcessorFast.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*local_dir_use_symlinks.*deprecated.*",
    category=UserWarning,
)

# -----------------------------
# Config (tuned for NVIDIA H100)
# -----------------------------
MODEL_ID = os.getenv(
    "MODEL_ID",
    "dataautogpt3/OpenDalle",
)

# Prefer BF16 on H100 (good speed + stability)
DTYPE_STR = os.getenv("TORCH_DTYPE", "bf16").lower()  # "bf16" or "fp16"
TORCH_DTYPE = torch.bfloat16 if DTYPE_STR in ("bf16", "bfloat16") else torch.float16

# Concurrency guard: 1 by default for single-GPU stability
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))

# Default generation params (safe-ish defaults; override per request)
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "20"))
DEFAULT_GUIDANCE = float(os.getenv("DEFAULT_GUIDANCE", "7.0"))

# Size policy
ALLOWED_SIZES_ENV = os.getenv("ALLOWED_SIZES", "512x512,768x768,1024x1024")
ALLOWED_SIZES = {s.strip() for s in ALLOWED_SIZES_ENV.split(",") if s.strip()}
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(1024 * 1024)))  # 1MP by default
REQUIRE_MULTIPLE_OF = int(os.getenv("REQUIRE_MULTIPLE_OF", "8"))

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="txt2image (H100)")

bearer_scheme = HTTPBearer()

# GPU semaphore (prevents OOM under load)
gpu_sem = asyncio.Semaphore(MAX_CONCURRENT)

# Global pipeline
pipe: Optional[AutoPipelineForText2Image] = None
last_used_at: float = time.time()
idle_monitor_task: Optional[asyncio.Task] = None
IDLE_UNLOAD_SECONDS = int(os.getenv("IDLE_UNLOAD_SECONDS", "3600"))
IDLE_MONITOR_INTERVAL_SECONDS = float(os.getenv("IDLE_MONITOR_INTERVAL_SECONDS", "1"))


# -----------------------------
# Request models
# -----------------------------
class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    n: int = Field(default=1, ge=1, le=8)
    size: str = Field(default="1024x1024")
    steps: int = Field(default=DEFAULT_STEPS, ge=1, le=150)
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE, ge=0.0, le=30.0)
    seed: Optional[int] = Field(default=None, ge=0)
    negative_prompt: Optional[str] = None
    response_format: Literal["b64_json"] = "b64_json"  # OpenAI-compatible subset


# -----------------------------
# Helpers
# -----------------------------
def parse_size(size_str: str) -> Tuple[int, int]:
    try:
        w_str, h_str = size_str.lower().split("x")
        w, h = int(w_str), int(h_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid size format. Use 'WxH' (e.g. 1024x1024).")

    if w <= 0 or h <= 0:
        raise HTTPException(status_code=400, detail="Width/height must be positive integers.")

    # Optional strict whitelist (recommended in prod)
    if ALLOWED_SIZES and size_str not in ALLOWED_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported size. Allowed: {sorted(ALLOWED_SIZES)}",
        )

    # Guard rails
    if w * h > MAX_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Requested size too large (w*h={w*h}). Max pixels: {MAX_PIXELS}.",
        )

    # Many diffusion models expect multiples of 8 (sometimes 16/64 depending on VAE)
    if (w % REQUIRE_MULTIPLE_OF) != 0 or (h % REQUIRE_MULTIPLE_OF) != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Width/height must be multiples of {REQUIRE_MULTIPLE_OF}.",
        )

    return w, h


def validate_bearer(credentials: HTTPAuthorizationCredentials) -> None:
    expected = os.getenv("OPENAI_API_KEY")
    if not expected:
        # Fail closed: you can set OPENAI_API_KEY in the env
        raise HTTPException(status_code=500, detail="Server misconfigured: OPENAI_API_KEY not set.")

    token = credentials.credentials or ""
    if not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    return torch.Generator(device="cuda").manual_seed(int(seed))


def encode_image_b64(img, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def generate_images(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    n: int,
    seed: Optional[int],
    negative_prompt: Optional[str],
) -> List:
    global pipe
    if pipe is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")

    gen = make_generator(seed)

    # For H100: bf16 + SDPA/Flash attention path (torch >= 2 recommended)
    # inference_mode + autocast reduces overhead & memory.
    with torch.inference_mode(), torch.autocast("cuda", dtype=TORCH_DTYPE):
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=n,
            generator=gen,
        )

    return out.images


def load_pipeline() -> None:
    global pipe

    # Perf toggles for Ampere/Hopper
    torch.backends.cuda.matmul.allow_tf32 = True  # safe speedup for matmul
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Load pipeline and use CPU if CUDA is not available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        # variant="fp16",  # uncomment if your repo has fp16 variant; for bf16 often not needed
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    # Prefer PyTorch SDPA (Flash Attention) when available
    # If you have xformers, you can enable it; but on H100 SDPA is often excellent already.
    if os.getenv("ENABLE_XFORMERS", "0") == "1":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            # Don't crash the server if xformers isn't available
            print(f"[WARN] xformers not enabled: {e}")

    # Optional: compile (can improve throughput, but increases startup time & memory)
    # Recommended only if you run long-lived server.
    if os.getenv("TORCH_COMPILE", "0") == "1":
        try:
            pipe.unet = torch.compile(pipe.unet, mode="max-autotune")
            print("[INFO] torch.compile enabled on UNet")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")


def unload_pipeline() -> None:
    global pipe
    if pipe is None:
        return

    try:
        pipe.to("cpu")
    except Exception:
        pass
    del pipe
    pipe = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    print("[INFO] pipeline unloaded from GPU due to idleness")


async def ensure_pipe_loaded() -> None:
    global pipe
    if pipe is None:
        print("[INFO] loading pipeline after idle unload")
        load_pipeline()


async def idle_unload_loop() -> None:
    global last_used_at
    while True:
        if IDLE_UNLOAD_SECONDS <= 0:
            await asyncio.sleep(60)
            continue
        poll_interval = max(0.1, min(IDLE_MONITOR_INTERVAL_SECONDS, float(IDLE_UNLOAD_SECONDS)))
        await asyncio.sleep(poll_interval)
        idle_for = time.time() - last_used_at
        if idle_for < IDLE_UNLOAD_SECONDS:
            continue
        async with gpu_sem:
            # Re-check after acquiring the semaphore to avoid races.
            idle_for = time.time() - last_used_at
            if idle_for >= IDLE_UNLOAD_SECONDS:
                unload_pipeline()
                # Avoid repeated unload attempts while still idle.
                last_used_at = time.time()

# -----------------------------
# Startup: init pipeline
# -----------------------------
@app.on_event("startup")
def startup() -> None:
    global idle_monitor_task, last_used_at

    load_pipeline()
    last_used_at = time.time()
    idle_monitor_task = asyncio.create_task(idle_unload_loop())

    # Warmup (small, fast)
    if os.getenv("WARMUP", "1") == "1":
        try:
            _ = pipe(
                prompt="warmup",
                width=512,
                height=512,
                num_inference_steps=2,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            )
            torch.cuda.synchronize()
            print("[INFO] warmup done")
        except Exception as e:
            print(f"[WARN] warmup failed: {e}")


@app.on_event("shutdown")
async def shutdown() -> None:
    global idle_monitor_task
    if idle_monitor_task is not None:
        idle_monitor_task.cancel()
        try:
            await idle_monitor_task
        except asyncio.CancelledError:
            pass


# -----------------------------
# Routes
# -----------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "model_id": MODEL_ID, "dtype": str(TORCH_DTYPE), "max_concurrent": MAX_CONCURRENT}



@app.post("/v1/images/generations")
async def create_image_generation(
    req: GenerationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    global last_used_at
    validate_bearer(credentials)

    width, height = parse_size(req.size)

    async with gpu_sem:
        await ensure_pipe_loaded()
        last_used_at = time.time()
        imgs = await generate_images(
            prompt=req.prompt,
            width=width,
            height=height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            n=req.n,
            seed=req.seed,
            negative_prompt=req.negative_prompt,
        )
        last_used_at = time.time()

    data = [{"b64_json": encode_image_b64(img, fmt="PNG")} for img in imgs]
    return JSONResponse({"created": int(time.time()), "data": data})


if __name__ == "__main__":
    import uvicorn

    # For a single H100: prefer 1 worker (multiple workers => multiple pipeline copies => VRAM blowup)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
