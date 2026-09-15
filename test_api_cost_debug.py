"""
test_api_cost_debug.py

Diagnostic for the API usage/cost tracking feature (tools.py):
- compute_api_cost() produces correct USD cost for known models
- compute_api_cost() degrades gracefully (returns None, no exception) for an unknown model
- _get_cached_usd_to_eur_rate() only calls get_exchange_rate once per day
- log_api_call() never raises even if the Sheets write fails

Uses fabricated model names/token counts only. Run directly: python test_api_cost_debug.py
"""

from unittest.mock import patch
import tools


def test_known_model_cost_groq():
    cost = tools.compute_api_cost("openai/gpt-oss-20b", input_tokens=1_000_000, output_tokens=1_000_000)
    expected = round(0.075 + 0.30, 6)
    assert cost == expected, f"Expected {expected}, got {cost}"
    print(f"PASS: gpt-oss-20b cost = {cost} USD for 1M/1M tokens")


def test_known_model_cost_claude():
    cost = tools.compute_api_cost("claude-sonnet-4-5", input_tokens=500_000, output_tokens=200_000)
    expected = round((0.5 * 3.0) + (0.2 * 15.0), 6)
    assert cost == expected, f"Expected {expected}, got {cost}"
    print(f"PASS: claude-sonnet-4-5 cost = {cost} USD for 500K/200K tokens")


def test_unknown_model_returns_none_no_crash():
    cost = tools.compute_api_cost("some-future-model-not-in-table", input_tokens=1000, output_tokens=1000)
    assert cost is None, f"Expected None for unknown model, got {cost}"
    print("PASS: unknown model returns None instead of raising")


def test_fx_rate_cached_once_per_day():
    tools._fx_cache["date"] = None
    tools._fx_cache["rate"] = None

    with patch("tools.get_exchange_rate", return_value=0.92) as mock_fx:
        rate1 = tools._get_cached_usd_to_eur_rate()
        rate2 = tools._get_cached_usd_to_eur_rate()
        rate3 = tools._get_cached_usd_to_eur_rate()

    assert rate1 == 0.92 and rate2 == 0.92 and rate3 == 0.92
    assert mock_fx.call_count == 1, f"Expected 1 call to get_exchange_rate, got {mock_fx.call_count}"
    print("PASS: FX rate fetched once and reused across repeated calls within the same day")


def test_log_api_call_never_raises_on_sheets_failure():
    with patch("tools.get_api_usage_worksheet", side_effect=Exception("simulated Sheets outage")):
        try:
            tools.log_api_call("Groq", "openai/gpt-oss-20b", 1000, 500)
            print("PASS: log_api_call swallows a Sheets failure instead of raising")
        except Exception as e:
            raise AssertionError(f"log_api_call raised instead of degrading gracefully: {e}")


if __name__ == "__main__":
    test_known_model_cost_groq()
    test_known_model_cost_claude()
    test_unknown_model_returns_none_no_crash()
    test_fx_rate_cached_once_per_day()
    test_log_api_call_never_raises_on_sheets_failure()
    print("\nAll API usage/cost tests passed.")
