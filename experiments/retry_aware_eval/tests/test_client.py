from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from retry_eval.client import OpenRouterClient, OpenRouterResponseError
from retry_eval.domain import CompletionRequest


class OpenRouterClientTests(unittest.TestCase):
    @patch("retry_eval.client.urlopen")
    def test_empty_visible_output_keeps_raw_usage_on_exception(self, urlopen):
        raw = {
            "id": "generation-id",
            "model": "openai/gpt-oss-20b",
            "provider": "example-provider",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None, "reasoning": "internal reasoning"},
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 512,
                "completion_tokens_details": {"reasoning_tokens": 500},
                "cost": 0.003,
            },
        }
        http_response = MagicMock()
        http_response.__enter__.return_value.status = 200
        http_response.__enter__.return_value.read.return_value = json.dumps(raw).encode()
        urlopen.return_value = http_response
        request = CompletionRequest(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "Solve it."}],
        )

        with self.assertRaises(OpenRouterResponseError) as raised:
            OpenRouterClient("sk-test").complete(request)

        self.assertIn("finish_reason='length'", str(raised.exception))
        self.assertEqual(raised.exception.response.raw_response, raw)
        self.assertEqual(raised.exception.response.output_tokens, 512)
        self.assertEqual(raised.exception.response.cost_usd, 0.003)
        self.assertEqual(raised.exception.response.provider, "example-provider")
        self.assertEqual(raised.exception.response.response_id, "generation-id")
        self.assertEqual(raised.exception.response.http_status, 200)
        self.assertEqual(raised.exception.response.finish_reason, "length")
        self.assertEqual(raised.exception.response.reasoning_tokens, 500)
        self.assertEqual(raised.exception.response.usage, raw["usage"])


if __name__ == "__main__":
    unittest.main()
