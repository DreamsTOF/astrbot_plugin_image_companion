# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from astrbot_plugin_image_companion.image_runtime import ImageGenerationRuntime
from astrbot_plugin_image_companion.main import ImageCompanionExtensionAPI


class AgnesEndpointRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output_path = Path(directory.name) / "result.png"
        self.output_path.write_bytes(b"\x89PNG\r\n\x1a\nresult")
        service = SimpleNamespace(
            data_dir=directory.name,
            image_data_lock=asyncio.Lock(),
            image_setting=lambda _name, default=None: default,
            _private_companion_api=lambda: object(),
        )
        self.api = ImageCompanionExtensionAPI(service)
        self.owner = SimpleNamespace(
            data_dir=directory.name,
            _extract_json_payload=json.loads,
        )
        self.endpoint = {
            "name": "主线 API 2",
            "platform": "agnes",
            "base_url": "https://agnes.example.test/v1/images/generations",
            "api_key": "test-api-key",
            "model": "agnes-image-2.1-flash",
            "size": "1K",
            "ratio": "2:3",
            "timeout_seconds": 20,
        }

    async def run_endpoint(self, responses):
        session = MagicMock()
        session.__aenter__.return_value = session
        queued = []
        for status, body in responses:
            response = MagicMock(status=status, headers={"Content-Type": "application/json"})
            response.__aenter__.return_value = response
            response.text = AsyncMock(return_value=json.dumps(body))
            queued.append(response)
        session.post.side_effect = queued
        download = AsyncMock(return_value=(str(self.output_path), "ok"))
        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch.object(ImageGenerationRuntime, "_download_external_image_url", download),
            patch.object(ImageGenerationRuntime, "_append_photo_generation_http_exchange") as exchanges,
        ):
            result = await self.api.test_endpoint(self.owner, self.endpoint, "a green check mark")
        return result, session.post.call_args_list, download, exchanges.call_args_list

    async def test_successful_http_response_reaches_image_materialization(self) -> None:
        image_url = "https://images.example.test/result.png"
        result, calls, download, exchanges = await self.run_endpoint(
            [(200, {"data": [{"url": image_url}]})]
        )

        self.assertTrue(result["ok"], result["message"])
        self.assertEqual(result["image_path"], str(self.output_path))
        self.assertIn("Agnes Image 1K/2:3", result["message"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0], self.endpoint["base_url"])
        download.assert_awaited_once_with(image_url, session_key="image_companion:endpoint_test")
        self.assertEqual(exchanges[0].kwargs["attempt"], 1)
        self.assertEqual(exchanges[0].kwargs["response_status"], 200)

    async def test_missing_path_continues_to_next_candidate(self) -> None:
        candidates = [
            self.endpoint["base_url"],
            "https://agnes.example.test/backup/v1/images/generations",
        ]
        with patch.object(ImageGenerationRuntime, "_external_image_endpoint_candidates", return_value=candidates):
            result, calls, download, exchanges = await self.run_endpoint(
                [
                    (404, {"error": {"message": "not found"}}),
                    (200, {"data": [{"url": "https://images.example.test/result.png"}]}),
                ]
            )

        self.assertTrue(result["ok"], result["message"])
        self.assertEqual([call.args[0] for call in calls], candidates)
        download.assert_awaited_once()
        self.assertEqual([call.kwargs["attempt"] for call in exchanges], [1, 2])
        self.assertEqual([call.kwargs["response_status"] for call in exchanges], [404, 200])

    async def test_provider_error_is_returned_without_materializing_an_image(self) -> None:
        result, calls, download, exchanges = await self.run_endpoint(
            [(401, {"error": {"message": "invalid API key"}})]
        )

        self.assertFalse(result["ok"])
        self.assertIn("401", result["message"])
        self.assertEqual(result["image_path"], "")
        self.assertEqual(len(calls), 1)
        download.assert_not_awaited()
        self.assertEqual(exchanges[0].kwargs["attempt"], 1)
        self.assertEqual(exchanges[0].kwargs["response_status"], 401)
