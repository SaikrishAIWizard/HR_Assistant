import json
import logging
from datetime import date
from types import SimpleNamespace

import pytest

from veltrix_assistant.llm import (
    EvidenceBlock,
    LLMError,
    OpenAIChatProvider,
)
from veltrix_assistant.schemas.documents import Citation


def _evidence() -> tuple[EvidenceBlock, ...]:
    return (
        EvidenceBlock(
            citation=Citation(doc_id="POLICY-1", locator="Annual leave"),
            text="Employees may carry forward unused annual leave.",
            doc_type="policy",
            regions=("GLOBAL",),
            effective_date=date(2025, 1, 1),
            upcoming=False,
        ),
    )


def test_openai_provider_sends_grounded_request_and_logs_corpus_tokens(
    monkeypatch,
    caplog,
) -> None:
    calls: list[dict[str, object]] = []
    model_response = {
        "status": "answered",
        "answer": "Unused annual leave may be carried forward.",
        "citations": [{"doc_id": "POLICY-1", "locator": "Annual leave"}],
    }

    def create_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(model_response))
                )
            ]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_completion)
        )
    )
    monkeypatch.setattr(
        "veltrix_assistant.llm.OpenAI",
        lambda **_: fake_client,
    )
    provider = OpenAIChatProvider(
        api_key="test-key",
        model="test-model",
        timeout_seconds=10,
    )

    with caplog.at_level(logging.INFO, logger="veltrix_assistant.llm"):
        answer = provider.generate_answer(
            "How much leave can E1001 carry forward? Email vijay@example.test, "
            "phone +1 415 555 0112, as of 2026-11-01.",
            _evidence(),
            reference_date=date(2026, 11, 1),
            corpus_tokens=23,
        )

    assert answer.answer == "Unused annual leave may be carried forward."
    assert answer.citations[0].doc_id == "POLICY-1"
    assert "llm_call corpus_tokens=23" in caplog.text
    request = calls[0]
    assert request["temperature"] == 0
    messages = request["messages"]
    assert isinstance(messages, list)
    user_message = messages[1]
    assert isinstance(user_message, dict)
    content = user_message["content"]
    assert isinstance(content, str)
    assert "2026-11-01" in content
    assert "E1001" not in content
    assert "vijay@example.test" not in content
    assert "+1 415 555 0112" not in content
    assert "carry forward unused annual leave" in content


def test_openai_provider_rejects_citations_outside_the_supplied_evidence(
    monkeypatch,
) -> None:
    response = {
        "status": "answered",
        "answer": "Unsupported answer.",
        "citations": [{"doc_id": "RESTRICTED", "locator": "Secret section"}],
    }
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=json.dumps(response))
                        )
                    ]
                )
            )
        )
    )
    monkeypatch.setattr(
        "veltrix_assistant.llm.OpenAI",
        lambda **_: fake_client,
    )
    provider = OpenAIChatProvider(
        api_key="test-key",
        model="test-model",
        timeout_seconds=10,
    )

    with pytest.raises(LLMError, match="cited unavailable evidence"):
        provider.generate_answer(
            "Question",
            _evidence(),
            reference_date=date(2026, 11, 1),
            corpus_tokens=1,
        )


def test_openai_provider_requires_configured_api_key() -> None:
    provider = OpenAIChatProvider(
        api_key=None,
        model="test-model",
        timeout_seconds=10,
    )

    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        provider.generate_answer(
            "Question",
            _evidence(),
            reference_date=date(2026, 11, 1),
            corpus_tokens=1,
        )
