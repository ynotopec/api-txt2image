import asyncio
import base64
import io
import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import app
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from PIL import Image


class Flux2TextEncoderLoadingTests(unittest.TestCase):
    def setUp(self):
        self.original_values = (
            app.pipe,
            app.MODEL_ID,
            app.PIPELINE_CLASS,
            app.FLUX2_BASE_MODEL_ID,
            app.FLUX2_TEXT_ENCODER_SUBFOLDER,
        )
        app.pipe = None
        app.MODEL_ID = "ponpoke/flux2-klein-4b-uncensored-text-encoder"
        app.PIPELINE_CLASS = "flux2_klein"
        app.FLUX2_BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
        app.FLUX2_TEXT_ENCODER_SUBFOLDER = "text_encoder"

    def tearDown(self):
        (
            app.pipe,
            app.MODEL_ID,
            app.PIPELINE_CLASS,
            app.FLUX2_BASE_MODEL_ID,
            app.FLUX2_TEXT_ENCODER_SUBFOLDER,
        ) = self.original_values

    def test_uses_base_encoder_config_with_replacement_weights(self):
        encoder_config = object()
        generation_config = object()
        text_encoder = object()
        pipeline = MagicMock()
        pipeline.to.return_value = pipeline

        with (
            patch.object(
                app.AutoConfig, "from_pretrained", return_value=encoder_config
            ) as load_config,
            patch.object(
                app.GenerationConfig,
                "from_model_config",
                return_value=generation_config,
            ) as make_generation_config,
            patch.object(
                app.AutoModelForCausalLM,
                "from_pretrained",
                return_value=text_encoder,
            ) as load_encoder,
            patch.object(
                app.Flux2KleinPipeline, "from_pretrained", return_value=pipeline
            ) as load_pipeline,
        ):
            app.load_pipeline()

        load_config.assert_called_once()
        config_args, config_kwargs = load_config.call_args
        self.assertEqual(config_args, (app.FLUX2_BASE_MODEL_ID,))
        self.assertEqual(config_kwargs["subfolder"], "text_encoder")
        self.assertNotIn("torch_dtype", config_kwargs)

        load_encoder.assert_called_once()
        encoder_args, encoder_kwargs = load_encoder.call_args
        self.assertEqual(encoder_args, (app.MODEL_ID,))
        self.assertIs(encoder_kwargs["config"], encoder_config)
        self.assertIs(encoder_kwargs["generation_config"], generation_config)
        self.assertEqual(encoder_kwargs["subfolder"], "text_encoder")
        make_generation_config.assert_called_once_with(encoder_config)

        load_pipeline.assert_called_once()
        pipeline_args, pipeline_kwargs = load_pipeline.call_args
        self.assertEqual(pipeline_args, (app.FLUX2_BASE_MODEL_ID,))
        self.assertIs(pipeline_kwargs["text_encoder"], text_encoder)

    def test_custom_named_safetensors_checkpoint_is_loaded(self):
        encoder_config = object()
        generation_config = object()
        text_encoder = object()

        with tempfile.TemporaryDirectory() as snapshot_dir:
            checkpoint = Path(snapshot_dir, "uncensored-text-encoder.safetensors")
            checkpoint.touch()
            missing_standard_file = OSError(
                f"{app.MODEL_ID} does not appear to have a file named "
                "pytorch_model.bin or model.safetensors."
            )

            with (
                patch.object(
                    app.GenerationConfig,
                    "from_model_config",
                    return_value=generation_config,
                ) as make_generation_config,
                patch.object(
                    app.AutoModelForCausalLM,
                    "from_pretrained",
                    side_effect=[missing_standard_file, text_encoder],
                ) as load_encoder,
                patch.object(
                    app,
                    "snapshot_download",
                    return_value=snapshot_dir,
                ) as download_snapshot,
            ):
                result = app.load_text_encoder_weights(
                    app.MODEL_ID,
                    encoder_config,
                    {"torch_dtype": app.TORCH_DTYPE, "local_files_only": False},
                )

        self.assertIs(result, text_encoder)
        make_generation_config.assert_called_once_with(encoder_config)
        download_snapshot.assert_called_once()
        self.assertEqual(load_encoder.call_count, 2)
        fallback_args, fallback_kwargs = load_encoder.call_args
        self.assertTrue(fallback_args[0].startswith("/tmp/flux2-text-encoder-"))
        self.assertIs(fallback_kwargs["config"], encoder_config)
        self.assertIs(fallback_kwargs["generation_config"], generation_config)
        self.assertTrue(fallback_kwargs["local_files_only"])
        self.assertNotIn("subfolder", fallback_kwargs)


class ImageEditingTests(unittest.TestCase):
    @staticmethod
    def _png_bytes(color="red"):
        source = io.BytesIO()
        Image.new("RGB", (8, 8), color).save(source, format="PNG")
        return source.getvalue()

    def test_openai_multipart_edit_endpoint_returns_base64_image(self):
        source = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(source, format="PNG")
        result_image = Image.new("RGB", (16, 16), "blue")

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch.object(app, "ensure_pipe_loaded", new=AsyncMock()),
            patch.object(
                app, "edit_images", new=AsyncMock(return_value=[result_image])
            ) as edit,
        ):
            response = TestClient(app.app).post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer test-key"},
                files={"image": ("input.png", source.getvalue(), "image/png")},
                data={
                    "prompt": "make it blue",
                    "model": "ignored-for-compatibility",
                    "size": "512x512",
                    "response_format": "b64_json",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("created", response.json())
        encoded = response.json()["data"][0]["b64_json"]
        self.assertTrue(base64.b64decode(encoded).startswith(b"\x89PNG"))
        self.assertEqual(edit.await_args.kwargs["prompt"], "make it blue")
        self.assertEqual(edit.await_args.kwargs["width"], 512)

    def test_openwebui_array_style_image_field_is_accepted(self):
        result_image = Image.new("RGB", (16, 16), "blue")

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch.object(app, "ensure_pipe_loaded", new=AsyncMock()),
            patch.object(
                app, "edit_images", new=AsyncMock(return_value=[result_image])
            ) as edit,
        ):
            response = TestClient(app.app).post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer test-key"},
                files=[
                    ("image[]", ("input.png", self._png_bytes(), "image/png"))
                ],
                data={"prompt": "add a cat", "size": "512x512"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(edit.await_args.kwargs["prompt"], "add a cat")

    def test_missing_edit_image_returns_descriptive_bad_request(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            response = TestClient(app.app).post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer test-key"},
                data={"prompt": "add a cat", "size": "512x512"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("'image' or 'image[]'", response.json()["detail"])

    def test_uploaded_image_is_decoded_as_rgb(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (12, 8), (255, 0, 0, 128)).save(buffer, format="PNG")
        upload = UploadFile(filename="input.png", file=io.BytesIO(buffer.getvalue()))

        result = asyncio.run(app.decode_uploaded_image(upload))

        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (12, 8))

    def test_invalid_uploaded_image_returns_bad_request(self):
        upload = UploadFile(filename="input.txt", file=io.BytesIO(b"not an image"))

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(app.decode_uploaded_image(upload))

        self.assertEqual(raised.exception.status_code, 400)

    def test_edit_uses_image_to_image_pipeline_and_openai_parameters(self):
        original_pipe = app.pipe

        class TextToImagePipeline:
            def __call__(self, prompt):
                del prompt

        app.pipe = TextToImagePipeline()
        output_image = Image.new("RGB", (16, 16))

        class FakeEditPipeline:
            def __init__(self):
                self.kwargs = None

            def __call__(
                self,
                prompt,
                image,
                strength,
                num_inference_steps,
                guidance_scale,
                num_images_per_prompt,
                generator,
                negative_prompt=None,
            ):
                self.kwargs = locals()
                return SimpleNamespace(images=[output_image])

        edit_pipeline = FakeEditPipeline()
        try:
            with patch.object(
                app.AutoPipelineForImage2Image,
                "from_pipe",
                return_value=edit_pipeline,
            ) as convert_pipeline:
                result = asyncio.run(
                    app.edit_images(
                        image=Image.new("RGB", (8, 8)),
                        prompt="make it blue",
                        width=16,
                        height=16,
                        steps=10,
                        guidance_scale=3.5,
                        strength=0.6,
                        n=1,
                        seed=None,
                        negative_prompt="red",
                    )
                )
        finally:
            app.pipe = original_pipe

        convert_pipeline.assert_called_once()
        self.assertEqual(result, [output_image])
        self.assertEqual(edit_pipeline.kwargs["prompt"], "make it blue")
        self.assertEqual(edit_pipeline.kwargs["image"].size, (16, 16))
        self.assertEqual(edit_pipeline.kwargs["strength"], 0.6)
        self.assertEqual(edit_pipeline.kwargs["negative_prompt"], "red")

    def test_flux2_klein_style_unified_pipeline_is_used_directly(self):
        original_pipe = app.pipe
        output_image = Image.new("RGB", (16, 16))

        class UnifiedPipeline:
            def __init__(self):
                self.kwargs = None

            def __call__(
                self,
                image=None,
                prompt=None,
                width=None,
                height=None,
                num_inference_steps=4,
                guidance_scale=1.0,
                num_images_per_prompt=1,
                generator=None,
                max_sequence_length=512,
            ):
                self.kwargs = locals()
                return SimpleNamespace(images=[output_image])

        unified_pipeline = UnifiedPipeline()
        app.pipe = unified_pipeline
        try:
            with patch.object(
                app.AutoPipelineForImage2Image, "from_pipe"
            ) as convert_pipeline:
                result = asyncio.run(
                    app.edit_images(
                        image=Image.new("RGB", (8, 8)),
                        prompt="change the background",
                        width=16,
                        height=16,
                        steps=4,
                        guidance_scale=1.0,
                        strength=0.75,
                        n=1,
                        seed=None,
                        negative_prompt=None,
                    )
                )
        finally:
            app.pipe = original_pipe

        convert_pipeline.assert_not_called()
        self.assertEqual(result, [output_image])
        self.assertEqual(unified_pipeline.kwargs["prompt"], "change the background")
        self.assertEqual(unified_pipeline.kwargs["image"].size, (16, 16))


if __name__ == "__main__":
    unittest.main()
