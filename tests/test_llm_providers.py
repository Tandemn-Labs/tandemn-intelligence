"""Unit tests for koi.llm.providers.build_model — env resolution + provider selection."""
import pytest

from koi.llm.providers import build_model


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "KOI_LLM_PROVIDER",
        "KOI_BASE_URL",
        "KOI_AGENT_MODEL",
        "KOI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_default_provider_is_openrouter(monkeypatch):
    monkeypatch.setenv("KOI_API_KEY", "sk-or-test")
    model = build_model()
    assert "openai" in type(model).__module__.lower()


def test_openrouter_accepts_openrouter_api_key_alias(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    model = build_model()
    assert "openai" in type(model).__module__.lower()


def test_openrouter_missing_key_raises():
    with pytest.raises(RuntimeError, match="KOI_API_KEY"):
        build_model()


def test_anthropic_provider(monkeypatch):
    monkeypatch.setenv("KOI_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    model = build_model()
    assert "anthropic" in type(model).__module__.lower()


def test_anthropic_missing_key_raises(monkeypatch):
    monkeypatch.setenv("KOI_LLM_PROVIDER", "anthropic")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_model()


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("KOI_LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError, match="unknown KOI_LLM_PROVIDER"):
        build_model()


def test_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("KOI_API_KEY", "env-key")
    monkeypatch.setenv("KOI_AGENT_MODEL", "env-model")
    model = build_model(api_key="explicit-key", model_id="deepseek/deepseek-chat")
    assert model is not None
