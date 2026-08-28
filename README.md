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

Edit an existing image with the OpenAI-compatible multipart endpoint used by
Open WebUI:

```bash
curl "$BASE_URL/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F 'image=@input.png' \
  -F 'prompt=turn the daytime scene into a moonlit night' \
  -F 'size=1024x1024' \
  -F 'response_format=b64_json'
```

The endpoint accepts the standard `image`, `prompt`, `model`, `n`, `size`, and
`response_format=b64_json` form fields. It also supports the optional local
controls `steps`, `guidance_scale`, `strength`, `seed`, and `negative_prompt`.
Parameters that are not implemented by the selected Diffusers pipeline are
ignored; in particular, FLUX.2 Klein uses its native reference-image editing
flow and does not expose `strength`.
The configured Diffusers checkpoint must have an image-to-image counterpart;
otherwise the API returns a descriptive HTTP 400 response. Uploaded images are
limited to `MAX_UPLOAD_BYTES` (20 MiB by default).

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

This configuration supports the `/v1/images/edits` endpoint too. FLUX.2 Klein
uses the same `Flux2KleinPipeline` for text generation and reference-image
editing, so no second checkpoint is loaded. For example, after starting the
service with the configuration above:

```bash
curl "$BASE_URL/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F 'image=@input.png' \
  -F 'prompt=replace the background with a snowy mountain landscape' \
  -F 'model=black-forest-labs/FLUX.2-klein-4B' \
  -F 'size=1024x1024'
```

The `model` form value is accepted for OpenAI/Open WebUI compatibility; this
single-model server always runs the checkpoint configured by `MODEL_ID`.

The `black-forest-labs/FLUX.2-klein-4b-fp8` and `-nvfp4` repositories are
single-file quantized transformer checkpoints, not complete Diffusers
pipelines. They contain auxiliary quantization tensors that Diffusers' FLUX.2
single-file converter does not currently handle. The API rejects these model
IDs with a clear HTTP 400 response instead of downloading the checkpoint and
failing during conversion. Use the complete model above, or a backend that
explicitly supports the quantized checkpoint format.

A replacement text-encoder repository can also be selected directly. For
example:

```env
MODEL_ID=ponpoke/flux2-klein-4b-uncensored-text-encoder
PIPELINE_CLASS=flux2_klein
FLUX2_BASE_MODEL_ID=black-forest-labs/FLUX.2-klein-base-4B
FLUX2_TEXT_ENCODER_SUBFOLDER=text_encoder
TORCH_DTYPE=bf16
DEFAULT_STEPS=50
DEFAULT_GUIDANCE=4.0
```

Because the component repository has no pipeline-level `model_index.json`, the
service loads its weights as the `text_encoder` and obtains the encoder
configuration, tokenizer, transformer, VAE, and scheduler from
`FLUX2_BASE_MODEL_ID`. This also supports component repositories whose minimal
`config.json` omits Transformers' `model_type` field. The base defaults to the
official, non-distilled 4B Klein base repository.

The replacement repository stores its Transformers checkpoint under its
`text_encoder/` directory rather than at the repository root. Its checkpoint
also uses a nonstandard filename. The loader first checks the configured
subfolder and, if no standard `model.safetensors` is present, discovers the
repository's single safetensors checkpoint and exposes it to Transformers under
the conventional filename without copying it. The subfolder setting defaults
to `text_encoder` and normally does not need to be changed.

The service also builds the encoder's generation configuration directly from
the base model configuration. This prevents recent Transformers versions from
trying to resolve a nonexistent `generation_config.json` in the temporary
checkpoint view.

The base checkpoint is not the four-step distilled checkpoint used in the
first example, so start with 50 steps and guidance 4.0 rather than the
distilled model's 4 steps and guidance 1.0.

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
