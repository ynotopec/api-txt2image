import unittest
from unittest.mock import MagicMock, patch

import app


class Flux2TextEncoderLoadingTests(unittest.TestCase):
    def setUp(self):
        self.original_values = (
            app.pipe,
            app.MODEL_ID,
            app.PIPELINE_CLASS,
            app.FLUX2_BASE_MODEL_ID,
        )
        app.pipe = None
        app.MODEL_ID = "ponpoke/flux2-klein-4b-uncensored-text-encoder"
        app.PIPELINE_CLASS = "flux2_klein"
        app.FLUX2_BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"

    def tearDown(self):
        (
            app.pipe,
            app.MODEL_ID,
            app.PIPELINE_CLASS,
            app.FLUX2_BASE_MODEL_ID,
        ) = self.original_values

    def test_uses_base_encoder_config_with_replacement_weights(self):
        encoder_config = object()
        text_encoder = object()
        pipeline = MagicMock()
        pipeline.to.return_value = pipeline

        with (
            patch.object(
                app.AutoConfig, "from_pretrained", return_value=encoder_config
            ) as load_config,
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

        load_pipeline.assert_called_once()
        pipeline_args, pipeline_kwargs = load_pipeline.call_args
        self.assertEqual(pipeline_args, (app.FLUX2_BASE_MODEL_ID,))
        self.assertIs(pipeline_kwargs["text_encoder"], text_encoder)


if __name__ == "__main__":
    unittest.main()
