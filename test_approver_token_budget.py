# test_approver_token_budget.py
from unittest.mock import MagicMock
import pipeline


def make_gathered_data(n_calls, chars_per_value):
    """Fake gathered_data big enough to blow past Groq's TPM limit if sent uncapped."""
    big_value = "x" * chars_per_value
    return [
        {
            "tool": f"fake_tool_{i}",
            "args": {"ticker": "TEST"},
            "result": {f"field_{j}": big_value for j in range(5)}
        }
        for i in range(n_calls)
    ]


def _run_with_mocks(gathered_data, draft):
    captured = {}

    def fake_call_groq_with_retry(client, **kwargs):
        captured["messages"] = kwargs["messages"]
        captured["max_tokens"] = kwargs["max_tokens"]
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = '{"approved": true, "issues": []}'
        return fake_response

    pipeline.extract_claims = lambda draft: []  # skip the real Claude call
    pipeline.call_groq_with_retry = fake_call_groq_with_retry

    verdict = pipeline.run_approver(gathered_data, draft)
    return verdict, captured


def test_large_gathered_data_gets_truncated_under_budget():
    gathered_data = make_gathered_data(n_calls=15, chars_per_value=2000)  # way past 1200/key alone
    draft = "MSFT trades at a P/E of 27.5x versus a peer median of 82.1x. " * 20

    verdict, captured = _run_with_mocks(gathered_data, draft)

    prompt_text = captured["messages"][0]["content"]
    requested_tokens = pipeline.estimate_tokens(prompt_text) + captured["max_tokens"]

    assert requested_tokens <= pipeline.GROQ_APPROVER_TOKEN_BUDGET, (
        f"Approver still requested ~{requested_tokens} tokens, over the "
        f"{pipeline.GROQ_APPROVER_TOKEN_BUDGET}-token safety budget"
    )
    assert verdict["approved"] is True
    print(f"PASS: large gathered_data truncated to ~{requested_tokens} tokens (budget {pipeline.GROQ_APPROVER_TOKEN_BUDGET})")


def test_small_gathered_data_is_not_truncated():
    gathered_data = make_gathered_data(n_calls=1, chars_per_value=50)
    draft = "MSFT trades at a P/E of 27.5x versus a peer median of 82.1x."

    verdict, captured = _run_with_mocks(gathered_data, draft)

    prompt_text = captured["messages"][0]["content"]
    assert "[truncated" not in prompt_text
    assert captured["max_tokens"] == pipeline.GROQ_APPROVER_MAX_TOKENS
    print("PASS: small gathered_data goes through untouched at the default per-key limit and max_tokens")


def test_reproduces_original_413_scenario_without_the_guard():
    """Sanity check: confirms this fixture really would have exceeded Groq's 8000-token
    TPM cap before this fix — the shape of the error Bernardo actually hit."""
    gathered_data = make_gathered_data(n_calls=15, chars_per_value=2000)
    draft = "MSFT trades at a P/E of 27.5x versus a peer median of 82.1x. " * 20

    uncapped_prompt = pipeline.APPROVER_PROMPT_TEMPLATE.format(
        gathered_data=pipeline.summarize_for_prompt(gathered_data),  # default per_key_limit=1200
        draft=draft
    )
    uncapped_requested = pipeline.estimate_tokens(uncapped_prompt) + pipeline.GROQ_APPROVER_MAX_TOKENS
    assert uncapped_requested > pipeline.GROQ_APPROVER_TPM_LIMIT, (
        "fixture doesn't actually reproduce an over-limit request — adjust n_calls/chars_per_value"
    )
    print(f"PASS: fixture reproduces the failure mode (~{uncapped_requested} tokens uncapped)")


test_large_gathered_data_gets_truncated_under_budget()
test_small_gathered_data_is_not_truncated()
test_reproduces_original_413_scenario_without_the_guard()
