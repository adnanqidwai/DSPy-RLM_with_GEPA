from rlm_gepa_retrieval.openai_compatible import OpenAICompatibleConfig, resolve_openai_compatible_config


def test_dspy_kwargs_prefix_openai_compatible_model_for_litellm() -> None:
    model = "mistralai/" + "Mini" + "stral-3-14B-Instruct-2512"
    config = OpenAICompatibleConfig(
        model_name=model,
        api_key="test-key",
        api_base="https://api.openai.com/v1",
    )

    kwargs = config.to_dspy_lm_kwargs(max_tokens=16, temperature=0)

    assert kwargs["model"] == f"openai/{model}"


def test_dspy_kwargs_do_not_double_prefix_openai_model() -> None:
    config = OpenAICompatibleConfig(model_name="openai/gpt-4o-mini", api_key="test-key", api_base=None)

    kwargs = config.to_dspy_lm_kwargs(max_tokens=16, temperature=0)

    assert kwargs["model"] == "openai/gpt-4o-mini"


def test_extra_headers_can_be_supplied_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_EXTRA_HEADERS_JSON", '{"X-Test": "value"}')

    config = resolve_openai_compatible_config("test-model", api_key="test-key", api_base="https://api.openai.com/v1")

    assert config.to_dspy_lm_kwargs(max_tokens=16, temperature=0)["extra_headers"] == {"X-Test": "value"}
