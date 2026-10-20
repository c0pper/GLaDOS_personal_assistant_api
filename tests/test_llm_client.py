from src.config import Config
from src.services.llm_client import fallback_routing


def test_fallbacks_follow_the_primary_model():
    assert fallback_routing("some/model") == {"models": ["some/model", *Config.LLM_FALLBACK_MODELS]}


def test_primary_model_is_not_repeated_as_fallback():
    primary = Config.LLM_FALLBACK_MODELS[0]

    models = fallback_routing(primary)["models"]

    assert models[0] == primary
    assert models.count(primary) == 1
