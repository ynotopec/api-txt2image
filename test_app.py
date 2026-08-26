import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import app


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


if __name__ == "__main__":
    unittest.main()
