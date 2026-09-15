# test_claude_retry_debug.py
from unittest.mock import MagicMock
import httpx
from anthropic import RateLimitError, APIStatusError, APIConnectionError
from tools import call_claude_with_retry

def fake_request():
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")

def test_succeeds_after_rate_limit():
    client = MagicMock()
    resp = httpx.Response(429, headers={"retry-after": "0"}, request=fake_request())
    client.messages.create.side_effect = [
        RateLimitError("rate limited", response=resp, body=None),
        "SUCCESS"
    ]
    result = call_claude_with_retry(client, model="x", max_tokens=10, messages=[])
    assert result == "SUCCESS"
    assert client.messages.create.call_count == 2
    print("PASS: recovers after one rate-limit retry")

def test_succeeds_after_server_error():
    client = MagicMock()
    resp = httpx.Response(503, request=fake_request())
    client.messages.create.side_effect = [
        APIStatusError("overloaded", response=resp, body=None),
        "SUCCESS"
    ]
    result = call_claude_with_retry(client, model="x", max_tokens=10, messages=[])
    assert result == "SUCCESS"
    print("PASS: recovers after one 5xx retry")

def test_does_not_retry_4xx_other_than_429():
    client = MagicMock()
    resp = httpx.Response(400, request=fake_request())
    client.messages.create.side_effect = APIStatusError("bad request", response=resp, body=None)
    try:
        call_claude_with_retry(client, model="x", max_tokens=10, messages=[])
        assert False, "should have raised"
    except APIStatusError:
        print("PASS: 400 is not retried, raises immediately")

def test_gives_up_after_max_retries():
    client = MagicMock()
    resp = httpx.Response(429, headers={"retry-after": "0"}, request=fake_request())
    client.messages.create.side_effect = RateLimitError("rate limited", response=resp, body=None)
    try:
        call_claude_with_retry(client, max_retries=2, model="x", max_tokens=10, messages=[])
        assert False, "should have raised after exhausting retries"
    except RateLimitError:
        print(f"PASS: raises after exhausting retries ({client.messages.create.call_count} attempts)")

test_succeeds_after_rate_limit()
test_succeeds_after_server_error()
test_does_not_retry_4xx_other_than_429()
test_gives_up_after_max_retries()
