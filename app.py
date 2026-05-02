# /home/ailab/api-txt2image/app.py
import os
import io
import time
import base64
import asyncio
import secrets
import warnings
import inspect
import gc
from typing import Optional, Literal, List, Tuple, Set

import torch
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from diffusers import AutoPipelineForText2Image, FluxPipeline

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
# Config
# -----------------------------
MODEL_ID = os.getenv("MODEL_ID", "dataautogpt3/OpenDalle")

DTYPE_STR = os.getenv("TORCH_DTYPE", "bf16").lower()
TORCH_DTYPE = torch.bfloat16 if DTYPE_STR in ("bf16", "bfloat16") else torch.float16

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))

DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "20"))
DEFAULT_GUIDANCE = float(os.getenv("DEFAULT_GUIDANCE", "7.0"))
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
PIPELINE_CLASS = os.getenv("PIPELINE_CLASS", "auto_t2i").strip().lower()

ALLOWED_SIZES_ENV = os.getenv("ALLOWED_SIZES", "512x512,768x768,1024x1024")
ALLOWED_SIZES = {s.strip() for s in ALLOWED_SIZES_ENV.split(",") if s.strip()}
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(1024 * 1024)))
REQUIRE_MULTIPLE_OF = int(os.getenv("REQUIRE_MULTIPLE_OF", "8"))

# Idle GPU memory management
# <= 0 disables idle unload.
IDLE_UNLOAD_SECONDS = int(os.getenv("IDLE_UNLOAD_SECONDS", "450"))
IDLE_MONITOR_INTERVAL_SECONDS = float(os.getenv("IDLE_MONITOR_INTERVAL_SECONDS", "30"))

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="txt2image")

bearer_scheme = HTTPBearer()
gpu_sem = asyncio.Semaphore(MAX_CONCURRENT)

pipe: Optional[AutoPipelineForText2Image] = None
last_used_at: float = time.time()
idle_monitor_task: Optional[asyncio.Task] = None


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
    response_format: Literal["b64_json"] = "b64_json"


# -----------------------------
# Helpers
# -----------------------------
def parse_size(size_str: str) -> Tuple[int, int]:
    try:
        w_str, h_str = size_str.lower().split("x")
        w, h = int(w_str), int(h_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid size format. Use 'WxH', e.g. 1024x1024.")

    if w <= 0 or h <= 0:
        raise HTTPException(status_code=400, detail="Width/height must be positive integers.")

    if ALLOWED_SIZES and size_str not in ALLOWED_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported size. Allowed: {sorted(ALLOWED_SIZES)}",
        )

    if w * h > MAX_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Requested size too large: {w*h}. Max pixels: {MAX_PIXELS}.",
        )

    if (w % REQUIRE_MULTIPLE_OF) != 0 or (h % REQUIRE_MULTIPLE_OF) != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Width/height must be multiples of {REQUIRE_MULTIPLE_OF}.",
        )

    return w, h


def get_expected_api_keys() -> Set[str]:
    keys_env = os.getenv("OPENAI_API_KEYS", "")
    single_key_env = os.getenv("OPENAI_API_KEY", "")

    raw_keys: List[str] = []

    if keys_env:
        raw_keys.extend(keys_env.replace("\n", ",").split(","))

    if single_key_env:
        raw_keys.append(single_key_env)

    return {k.strip() for k in raw_keys if k.strip()}


def validate_bearer(credentials: HTTPAuthorizationCredentials) -> None:
    expected_keys = get_expected_api_keys()

    if not expected_keys:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: OPENAI_API_KEY or OPENAI_API_KEYS must be set.",
        )

    token = (credentials.credentials or "").strip()

    if not any(secrets.compare_digest(token, expected) for expected in expected_keys):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.Generator(device=device).manual_seed(int(seed))


def encode_image_b64(img, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -----------------------------
# Pipeline lifecycle
# -----------------------------
def load_pipeline() -> None:
    global pipe

    if pipe is not None:
        return

    torch.backends.cuda.matmul.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )

    local_files_only = os.getenv("LOCAL_FILES_ONLY", "0") == "1"
    cache_dir = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")

    if PIPELINE_CLASS == "flux":
        pipeline_loader = FluxPipeline
        resolved_pipeline_class = "flux"
    else:
        pipeline_loader = AutoPipelineForText2Image
        resolved_pipeline_class = "auto_t2i"

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "1"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        "[INFO] Loading model="
        f"'{MODEL_ID}' pipeline_class='{resolved_pipeline_class}' dtype='{TORCH_DTYPE}' "
        f"device='{device}' local_files_only={local_files_only} "
        f"cache_dir='{cache_dir or 'default'}' "
        f"hf_transfer={os.getenv('HF_HUB_ENABLE_HF_TRANSFER')}"
    )

    t0 = time.perf_counter()

    pipe = pipeline_loader.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        token=hf_token,
        local_files_only=local_files_only,
    ).to(device)

    pipe.set_progress_bar_config(disable=True)

    if os.getenv("ENABLE_XFORMERS", "0") == "1":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[INFO] xformers enabled")
        except Exception as e:
            print(f"[WARN] xformers not enabled: {e}")

    if os.getenv("TORCH_COMPILE", "0") == "1":
        try:
            if hasattr(pipe, "unet") and pipe.unet is not None:
                pipe.unet = torch.compile(pipe.unet, mode="max-autotune")
                print("[INFO] torch.compile enabled on UNet")
            else:
                print("[WARN] torch.compile skipped: pipeline has no UNet")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    print(f"[INFO] Pipeline loaded in {time.perf_counter() - t0:.1f}s on device='{device}'")


def unload_pipeline() -> None:
    """
    Fully unload the pipeline.

    Important:
    - Do NOT use pipe.to("cpu").
      It copies a huge model from GPU to CPU, burns CPU, fills RAM/swap,
      and can freeze the host.
    - We delete the pipeline and clear CUDA cache.
      The next request reloads it from disk/cache.
    """
    global pipe

    if pipe is None:
        return

    print("[INFO] unloading pipeline: deleting pipeline and clearing CUDA cache")
    t0 = time.perf_counter()

    try:
        del pipe
    finally:
        pipe = None

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    print(f"[INFO] pipeline fully unloaded in {time.perf_counter() - t0:.1f}s")


async def ensure_pipe_loaded() -> None:
    global pipe

    if pipe is None:
        print("[INFO] loading pipeline after idle unload")
        await asyncio.to_thread(load_pipeline)


async def idle_unload_loop() -> None:
    global last_used_at

    print(
        f"[INFO] idle unload monitor started: "
        f"IDLE_UNLOAD_SECONDS={IDLE_UNLOAD_SECONDS}, "
        f"IDLE_MONITOR_INTERVAL_SECONDS={IDLE_MONITOR_INTERVAL_SECONDS}"
    )

    while True:
        if IDLE_UNLOAD_SECONDS <= 0:
            await asyncio.sleep(60)
            continue

        poll_interval = max(
            1.0,
            min(float(IDLE_MONITOR_INTERVAL_SECONDS), float(IDLE_UNLOAD_SECONDS)),
        )

        await asyncio.sleep(poll_interval)

        if pipe is None:
            continue

        idle_for = time.time() - last_used_at

        if idle_for < IDLE_UNLOAD_SECONDS:
            continue

        async with gpu_sem:
            if pipe is None:
                continue

            idle_for = time.time() - last_used_at

            if idle_for >= IDLE_UNLOAD_SECONDS:
                print(f"[INFO] pipeline idle for {idle_for:.1f}s, unloading")
                await asyncio.to_thread(unload_pipeline)
                last_used_at = time.time()


# -----------------------------
# Generation
# -----------------------------
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

    call_kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": n,
        "generator": gen,
    }

    call_sig = inspect.signature(pipe.__call__)

    if negative_prompt is not None and "negative_prompt" in call_sig.parameters:
        call_kwargs["negative_prompt"] = negative_prompt

    if "max_sequence_length" in call_sig.parameters:
        call_kwargs["max_sequence_length"] = MAX_SEQUENCE_LENGTH

    def _run_pipeline():
        if torch.cuda.is_available():
            with torch.inference_mode(), torch.autocast("cuda", dtype=TORCH_DTYPE):
                return pipe(**call_kwargs)
        else:
            with torch.inference_mode():
                return pipe(**call_kwargs)

    out = await asyncio.to_thread(_run_pipeline)
    return out.images


# -----------------------------
# Startup / shutdown
# -----------------------------
@app.on_event("startup")
async def startup() -> None:
    global idle_monitor_task, last_used_at

    await asyncio.to_thread(load_pipeline)
    last_used_at = time.time()

    idle_monitor_task = asyncio.create_task(idle_unload_loop())

    if os.getenv("WARMUP", "0") == "1":
        try:
            print("[INFO] warmup started")

            def _warmup():
                _ = pipe(
                    prompt="warmup",
                    width=512,
                    height=512,
                    num_inference_steps=2,
                    guidance_scale=1.0,
                    num_images_per_prompt=1,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            await asyncio.to_thread(_warmup)
            last_used_at = time.time()
            print("[INFO] warmup done")
        except Exception as e:
            print(f"[WARN] warmup failed: {e}")
    else:
        print("[INFO] warmup skipped (set WARMUP=1 to enable)")


@app.on_event("shutdown")
async def shutdown() -> None:
    global idle_monitor_task

    if idle_monitor_task is not None:
        idle_monitor_task.cancel()
        try:
            await idle_monitor_task
        except asyncio.CancelledError:
            pass

    await asyncio.to_thread(unload_pipeline)


# -----------------------------
# Routes
# -----------------------------
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "model_id": MODEL_ID,
        "dtype": str(TORCH_DTYPE),
        "max_concurrent": MAX_CONCURRENT,
        "pipeline_loaded": pipe is not None,
        "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        "idle_monitor_interval_seconds": IDLE_MONITOR_INTERVAL_SECONDS,
    }


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

    return JSONResponse(
        {
            "created": int(time.time()),
            "data": data,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
