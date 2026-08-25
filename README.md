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

## Quantized Krea 2 Turbo

The `OzzyGT/Krea_2_Turbo_nunchaku_lite_nvfp4` repository can be selected through
the existing Diffusers loader. It contains 4-bit components, so the project
installs `bitsandbytes>=0.46.1` as part of its runtime dependencies. Recommended
settings from the model example are:

```env
MODEL_ID=OzzyGT/Krea_2_Turbo_nunchaku_lite_nvfp4
PIPELINE_CLASS=auto_t2i
TORCH_DTYPE=bf16
DEFAULT_STEPS=8
DEFAULT_GUIDANCE=0.0
```

Run `./install.sh` again after updating an existing checkout so the new
dependency is installed. The first generation request downloads and loads the
model unless pipeline preloading is enabled.

## FLUX.2 Klein 4B

Use the complete Diffusers repository with the explicit Klein loader:

```env
MODEL_ID=black-forest-labs/FLUX.2-klein-4b
PIPELINE_CLASS=flux2_klein
TORCH_DTYPE=bf16
DEFAULT_STEPS=4
DEFAULT_GUIDANCE=1.0
```

The `black-forest-labs/FLUX.2-klein-4b-fp8` and `-nvfp4` repositories are
single-file quantized transformer checkpoints, not complete Diffusers
pipelines. They contain auxiliary quantization tensors that Diffusers' FLUX.2
single-file converter does not currently handle. The API rejects these model
IDs with a clear HTTP 400 response instead of downloading the checkpoint and
failing during conversion. Use the complete model above, or a backend that
explicitly supports the quantized checkpoint format.

## Sana Sprint 0.6B

Sana Sprint declares `SanaSprintPipeline`, which Diffusers' generic
`AutoPipelineForText2Image` does not currently resolve. Select its explicit
loader through the environment:

```env
MODEL_ID=Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers
PIPELINE_CLASS=sana_sprint
TORCH_DTYPE=bf16
DEFAULT_STEPS=2
DEFAULT_GUIDANCE=0.0
```

The pipeline accepts the same image-generation endpoint and request shape as
the other supported models.

## Z-Image Turbo

Use the official Diffusers repository with its explicit `ZImagePipeline`
loader:

```env
MODEL_ID=Tongyi-MAI/Z-Image-Turbo
PIPELINE_CLASS=z_image
TORCH_DTYPE=bf16
DEFAULT_STEPS=9
DEFAULT_GUIDANCE=0.0
```

The `Comfy-Org/z_image_turbo` files under `split_files/diffusion_models` are
ComfyUI-native component checkpoints. In particular,
`z_image_turbo_nvfp4.safetensors` uses a packed NVFP4 representation that
Diffusers' Z-Image checkpoint converter cannot read; it is not a complete
Diffusers pipeline and NVIDIA ModelOpt does not make that file format
compatible. Run that exact checkpoint through ComfyUI instead. To avoid a long
download followed by an internal conversion traceback, the API rejects a
single-file `.safetensors` value with `PIPELINE_CLASS=z_image` and returns a
clear HTTP 400 response.

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
