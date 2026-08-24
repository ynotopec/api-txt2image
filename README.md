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

## Z-Image Turbo NVFP4

The Comfy-Org single-file NVFP4 checkpoint can be loaded with Diffusers'
explicit `ZImagePipeline` loader:

```env
MODEL_ID=https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_nvfp4.safetensors
PIPELINE_CLASS=z_image
TORCH_DTYPE=bf16
DEFAULT_STEPS=9
DEFAULT_GUIDANCE=0.0
```

Run `./install.sh` after updating so that NVIDIA ModelOpt is installed. NVFP4
execution requires an NVIDIA Blackwell GPU (for example, DGX Spark or an RTX
50-series card); it is not supported by H100 hardware. Use the `/resolve/`
download URL shown above rather than the Hugging Face `/blob/` page URL from a
browser. As with the other models, the checkpoint is fetched on the first
request unless startup preloading is enabled.

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
