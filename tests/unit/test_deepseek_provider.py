"""DeepSeekProvider puts the enable_thinking switch into the request body."""

import json

import httpx

from app.config.settings import Settings
from app.providers.deepseek import DeepSeekProvider


async def _send_and_capture(settings: Settings) -> dict:
    """Intercept the HTTP request with a mock transport and return its JSON body."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}}], "usage": {}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://llm.test")
    provider = DeepSeekProvider(settings, client=client)
    await provider.generate([{"role": "user", "content": "你好"}])
    await client.aclose()
    return captured


async def test_thinking_param_omitted_by_default() -> None:
    payload = await _send_and_capture(Settings(_env_file=None, deepseek_api_key="sk-test"))
    assert "enable_thinking" not in payload


async def test_thinking_can_be_turned_off() -> None:
    payload = await _send_and_capture(
        Settings(_env_file=None, deepseek_api_key="sk-test", llm_enable_thinking=False)
    )
    assert payload["enable_thinking"] is False


async def test_thinking_can_be_turned_on() -> None:
    payload = await _send_and_capture(
        Settings(_env_file=None, deepseek_api_key="sk-test", llm_enable_thinking=True)
    )
    assert payload["enable_thinking"] is True
