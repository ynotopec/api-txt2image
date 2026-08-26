import pytest

import app


class FakePipeline:
    def __init__(self):
        self.device = None
        self.progress_bar_disabled = False

    def to(self, device):
        self.device = device
        return self

    def set_progress_bar_config(self, *, disable):
        self.progress_bar_disabled = disable


@pytest.fixture(autouse=True)
def reset_pipeline(monkeypatch):
    monkeypatch.setattr(app, "pipe", None)
    monkeypatch.setattr(app.torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("ENABLE_XFORMERS", raising=False)
    monkeypatch.delenv("TORCH_COMPILE", raising=False)
    yield
    app.pipe = None


def test_z_image_uses_explicit_diffusers_loader(monkeypatch):
    calls = []
    fake_pipeline = FakePipeline()

    class FakeZImagePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return fake_pipeline

    monkeypatch.setattr(app, "PIPELINE_CLASS", "z_image")
    monkeypatch.setattr(app, "MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
    monkeypatch.setattr(app, "ZImagePipeline", FakeZImagePipeline)

    app.load_pipeline()

    assert calls[0][0] == "Tongyi-MAI/Z-Image-Turbo"
    assert calls[0][1]["torch_dtype"] is app.TORCH_DTYPE
    assert fake_pipeline.device == "cpu"
    assert fake_pipeline.progress_bar_disabled is True
    assert app.pipe is fake_pipeline


def test_z_image_rejects_single_file_checkpoint_before_loading(monkeypatch):
    class UnexpectedLoader:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            pytest.fail("from_pretrained must not run for a single-file Z-Image checkpoint")

    monkeypatch.setattr(app, "PIPELINE_CLASS", "z_image")
    monkeypatch.setattr(
        app,
        "MODEL_ID",
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
        "split_files/diffusion_models/z_image_turbo_nvfp4.safetensors",
    )
    monkeypatch.setattr(app, "ZImagePipeline", UnexpectedLoader)

    with pytest.raises(app.UnsupportedModelError, match="ComfyUI-native"):
        app.load_pipeline()

    assert app.pipe is None
