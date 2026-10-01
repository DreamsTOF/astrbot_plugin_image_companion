# -*- coding: utf-8 -*-
"""Online image routes must accept any model name and submit reference edits.

Covers the removal of local model whitelists and per-route reference-image
blocks for SenseNova, ModelScope and OpenAI-compatible endpoints.  All HTTP
calls run against mocked sessions, never real providers.
"""
from __future__ import annotations

import asyncio
import base64
import json
import struct
import tempfile
import unittest
import zlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from astrbot_plugin_image_companion.image_runtime import ImageGenerationRuntime

_SESSION_KEY = "image_companion:test"


def _png_bytes() -> bytes:
    width = height = 8
    raw = b"".join(b"\x00" + b"\xe0\x20\x20" * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _b64_png_data_url() -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes()).decode("ascii")


def _response(status: int, body: dict) -> MagicMock:
    response = MagicMock(status=status, headers={"Content-Type": "application/json"})
    response.__aenter__.return_value = response
    response.text = AsyncMock(return_value=json.dumps(body))
    return response


def _session(*posts: MagicMock, gets: tuple[MagicMock, ...] = ()) -> MagicMock:
    session = MagicMock()
    session.__aenter__.return_value = session
    session.post.side_effect = list(posts)
    if gets:
        session.get.side_effect = list(gets)
    return session


class ExternalRouteRestrictionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.reference = self.directory / "reference.png"
        self.reference.write_bytes(_png_bytes())
        self.service = SimpleNamespace(
            data_dir=directory.name,
            image_data_lock=asyncio.Lock(),
            image_setting=lambda _name, default=None: default,
            _private_companion_api=lambda: object(),
        )
        self.owner = SimpleNamespace(
            data_dir=directory.name,
            _extract_json_payload=json.loads,
        )
        self.owner._environment_now = lambda: datetime.now()
        self.runtime = ImageGenerationRuntime(self.service, self.owner)
        self.runtime.external_image_api_timeout_seconds = 60

    def endpoint(self, **overrides) -> dict:
        config = {
            "name": "test",
            "platform": "auto",
            "base_url": "",
            "api_key": "test-key",
            "model": "",
            "size": "1024x1024",
            "timeout_seconds": 60,
        }
        config.update(overrides)
        return config

    async def run_generation(self, endpoint: dict, *, reference: bool = False, image_size: str = "1024x1024"):
        with patch.object(ImageGenerationRuntime, "_append_photo_generation_http_exchange"):
            return await self.runtime._run_external_photo_generation_with_endpoint(
                endpoint,
                "a red circle on a blue background",
                session_key=_SESSION_KEY,
                reference_image_path=str(self.reference) if reference else "",
                reference_image_paths=(str(self.reference),) if reference else (),
                image_size=image_size,
            )

    # 1. 模型名不再受本地白名单限制 -------------------------------

    async def test_sensenova_accepts_u1_5_model(self) -> None:
        endpoint = self.endpoint(
            platform="auto",
            base_url="https://token.sensenova.cn/v1",
            model="sensenova-u1.5-fast",
        )
        self.runtime._apply_external_image_api_endpoint_runtime(endpoint)
        self.assertEqual(self.runtime._resolved_external_image_api_platform(), "sensenova")
        self.assertEqual(self.runtime._external_image_model_misconfiguration_note(), "")
        self.assertEqual(
            self.runtime._external_image_api_endpoint_unavailable_note(endpoint),
            "",
        )

    async def test_modelscope_accepts_official_model_ids(self) -> None:
        endpoint = self.endpoint(
            platform="modelscope",
            base_url="https://api-inference.modelscope.cn/",
            model="Qwen/Qwen-Image-2.1",
        )
        self.runtime._apply_external_image_api_endpoint_runtime(endpoint)
        self.assertEqual(self.runtime._external_image_model_misconfiguration_note(), "")
        self.assertEqual(
            self.runtime._external_image_api_endpoint_unavailable_note(endpoint),
            "",
        )

    async def test_openai_compatible_accepts_any_image_model_name(self) -> None:
        for model in ("sensenova-u1.5-fast", "qwen-image-edit", "my-custom-image-model"):
            endpoint = self.endpoint(
                platform="openai",
                base_url="https://proxy.example.test/v1",
                model=model,
            )
            self.runtime._apply_external_image_api_endpoint_runtime(endpoint)
            self.assertEqual(
                self.runtime._external_image_model_misconfiguration_note(),
                "",
                model,
            )

    # 6. 未来模型名同样不受限 ------------------------------------

    async def test_sensenova_route_is_decided_by_address_not_model_name(self) -> None:
        # 官方地址 + 任意模型名（哪怕以后不叫 sensenova）都走日日新协议。
        for model in ("sensenova-v3", "qwen-renamed-model", "future-image-model"):
            runtime = ImageGenerationRuntime(self.service, self.owner)
            runtime.external_image_api_platform = "auto"
            runtime.external_image_api_base_url = "https://token.sensenova.cn/v1"
            runtime.external_image_api_model = model
            self.assertEqual(
                runtime._resolved_external_image_api_platform(),
                "sensenova",
                model,
            )
            self.assertEqual(
                runtime._external_image_model_misconfiguration_note(),
                "",
                model,
            )
        # 显式选择平台时同样任意模型名可用。
        runtime = ImageGenerationRuntime(self.service, self.owner)
        runtime.external_image_api_platform = "sensenova"
        runtime.external_image_api_base_url = "https://gateway.example.test/v1"
        runtime.external_image_api_model = "whatever-renamed-model"
        self.assertEqual(runtime._resolved_external_image_api_platform(), "sensenova")

    async def test_model_names_never_hijack_platform_routing(self) -> None:
        # 模型名不再触发任何平台劫持：非官方地址一律按 OpenAI 兼容协议，
        # 需要其他协议就显式选择平台。
        cases = [
            ("qwen-image-next", "https://proxy.example.test/v1"),
            ("seedream-x9", "https://proxy.example.test/v1"),
            ("gemini-3-image", "https://proxy.example.test/v1"),
            ("qwen-image-next", ""),
            ("seedream-x9", ""),
        ]
        for model, base in cases:
            runtime = ImageGenerationRuntime(self.service, self.owner)
            runtime.external_image_api_platform = "auto"
            runtime.external_image_api_base_url = base
            runtime.external_image_api_model = model
            self.assertEqual(
                runtime._resolved_external_image_api_platform(),
                "openai",
                (model, base),
            )

    async def test_openai_multi_reference_capacity_ignores_model_name(self) -> None:
        capacity = self.runtime._external_image_endpoint_multi_reference_capacity(
            {
                "platform": "openai",
                "base_url": "https://proxy.example.test/v1",
                "model": "any-future-model",
            },
            5,
        )
        self.assertEqual(capacity, 5)
        self.assertEqual(
            self.runtime._external_image_endpoint_multi_reference_capacity(
                {
                    "platform": "openai",
                    "base_url": "https://proxy.example.test/v1",
                    "model": "any-future-model",
                },
                12,
            ),
            8,
        )

    async def test_modelscope_route_accepts_any_model_name(self) -> None:
        for model in ("Qwen/Qwen-Image-2.1", "AI-ModelScope/future-model", "qwen-image-next"):
            endpoint = self.endpoint(
                platform="modelscope",
                base_url="https://api-inference.modelscope.cn/",
                model=model,
            )
            runtime = ImageGenerationRuntime(self.service, self.owner)
            runtime._apply_external_image_api_endpoint_runtime(endpoint)
            self.assertEqual(runtime._resolved_external_image_api_platform(), "modelscope", model)
            self.assertEqual(
                runtime._external_image_model_misconfiguration_note(),
                "",
                model,
            )

    async def test_modelscope_base_wins_over_bailian_model_hijack(self) -> None:
        # 平台 auto + 魔搭地址 + 千问系模型名时，必须仍然走魔搭协议。
        runtime = ImageGenerationRuntime(self.service, self.owner)
        runtime.external_image_api_platform = "auto"
        runtime.external_image_api_base_url = "https://api-inference.modelscope.cn/"
        runtime.external_image_api_model = "qwen-image-future"
        self.assertEqual(runtime._resolved_external_image_api_platform(), "modelscope")

    async def test_sensenova_edit_submits_configured_model_verbatim(self) -> None:
        endpoint = self.endpoint(
            platform="sensenova",
            base_url="https://token.sensenova.cn/v1",
            model="sensenova-future-model",
        )
        session = _session(_response(200, {"data": [{"b64_json": _b64_png_data_url()}]}))
        with patch("aiohttp.ClientSession", return_value=session):
            outcome = await self.run_generation(endpoint, reference=True)

        self.assertTrue(outcome.image_path, outcome.note)
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "sensenova-future-model")

    # 7. 参考图统一归一化 ----------------------------------------

    def test_oversized_reference_is_scaled_proportionally(self) -> None:
        import io

        from PIL import Image as PILImage

        source = PILImage.new("RGB", (2400, 1000), (200, 30, 30))
        buffer = io.BytesIO()
        source.save(buffer, "JPEG")
        data, mime = ImageGenerationRuntime._downscale_reference_image_bytes(
            buffer.getvalue()
        )
        self.assertEqual(mime, "image/jpeg")
        with PILImage.open(io.BytesIO(data)) as scaled:
            self.assertEqual(scaled.size, (2048, 853))
        self.assertLess(len(data), len(buffer.getvalue()))

    def test_oversized_transparent_reference_stays_png(self) -> None:
        import io

        from PIL import Image as PILImage

        source = PILImage.new("RGBA", (3000, 500), (255, 0, 0, 128))
        buffer = io.BytesIO()
        source.save(buffer, "PNG")
        data, mime = ImageGenerationRuntime._downscale_reference_image_bytes(
            buffer.getvalue()
        )
        self.assertEqual(mime, "image/png")
        with PILImage.open(io.BytesIO(data)) as scaled:
            self.assertEqual(scaled.size, (2048, 341))
            self.assertEqual(scaled.mode, "RGBA")

    def test_within_limit_reference_passes_through_verbatim(self) -> None:
        original = _png_bytes()
        data, mime = ImageGenerationRuntime._downscale_reference_image_bytes(original)
        self.assertEqual(data, original)
        self.assertEqual(mime, "image/png")

    async def test_reference_data_url_is_normalized(self) -> None:
        import io

        from PIL import Image as PILImage

        oversized = self.directory / "oversized.png"
        source = PILImage.new("RGB", (2400, 1000), (10, 120, 240))
        source.save(oversized, "PNG")
        data_url = await self.runtime._reference_image_to_data_url(str(oversized))
        header, encoded = data_url.split(",", 1)
        self.assertEqual(header, "data:image/jpeg;base64")
        with PILImage.open(io.BytesIO(base64.b64decode(encoded))) as scaled:
            self.assertEqual(scaled.size, (2048, 853))

    async def test_modelscope_output_size_passes_through_verbatim(self) -> None:
        # 用户填 4096 就提交 4096：不本地钳制，服务端不支持就返回错误。
        endpoint = self.endpoint(
            platform="modelscope",
            base_url="https://api-inference.modelscope.cn/",
            model="Qwen/Qwen-Image-2.1",
            size="4096x4096",
        )
        session = _session(
            _response(200, {"task_id": "task-1"}),
            gets=(
                _response(
                    200,
                    {"task_status": "SUCCEED", "output_images": [_b64_png_data_url()]},
                ),
            ),
        )
        with patch("aiohttp.ClientSession", return_value=session):
            outcome = await self.run_generation(endpoint, image_size="")

        self.assertTrue(outcome.image_path, outcome.note)
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["size"], "4096x4096")

    # 2. SenseNova 尺寸不再强制映射旧档位 -------------------------

    def test_sensenova_size_passthrough_for_valid_sizes(self) -> None:
        for size in ("1024x1024", "2048x2048", "2048x1152", "512x512", "2720x1536"):
            self.assertEqual(self.runtime._sanitize_sensenova_image_size(size), size, size)

    def test_sensenova_size_maps_invalid_input(self) -> None:
        self.assertEqual(self.runtime._sanitize_sensenova_image_size("abc"), "2752x1536")
        self.assertEqual(self.runtime._sanitize_sensenova_image_size("1000x1000"), "2048x2048")
        mapped = self.runtime._sanitize_sensenova_image_size("4320x1024")
        self.assertIn(
            mapped,
            {"3072x1376", "2752x1536", "2496x1664", "2048x2048"},
        )

    # 3. SenseNova 参考图编辑 ------------------------------------

    async def test_sensenova_reference_edit_submits_json_images(self) -> None:
        endpoint = self.endpoint(
            platform="auto",
            base_url="https://token.sensenova.cn/v1",
            model="sensenova-u1.5-fast",
        )
        session = _session(_response(200, {"data": [{"b64_json": _b64_png_data_url()}]}))
        with patch("aiohttp.ClientSession", return_value=session):
            outcome = await self.run_generation(endpoint, reference=True)

        self.assertTrue(outcome.image_path, outcome.note)
        self.assertTrue(Path(outcome.image_path).exists())
        self.assertEqual(session.post.call_count, 1)
        submitted_url = session.post.call_args.args[0]
        self.assertTrue(submitted_url.endswith("/v1/images/edits"), submitted_url)
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "sensenova-u1.5-fast")
        self.assertEqual(payload["n"], 1)
        self.assertTrue(payload["images"][0]["image_url"].startswith("data:image/png;base64,"))

    # 4. 魔搭任务类型与参考图 ------------------------------------

    async def test_modelscope_poll_requires_image_generation_task_type(self) -> None:
        endpoint = self.endpoint(
            platform="modelscope",
            base_url="https://api-inference.modelscope.cn/",
            model="Qwen/Qwen-Image-2.1",
        )
        session = _session(
            _response(200, {"task_id": "task-1"}),
            gets=(
                _response(
                    200,
                    {"task_status": "SUCCEED", "output_images": [_b64_png_data_url()]},
                ),
            ),
        )
        with patch("aiohttp.ClientSession", return_value=session):
            outcome = await self.run_generation(endpoint)

        self.assertTrue(outcome.image_path, outcome.note)
        submit_headers = session.post.call_args.kwargs["headers"]
        self.assertEqual(submit_headers["X-ModelScope-Async-Mode"], "true")
        self.assertEqual(submit_headers["X-ModelScope-Task-Type"], "image_generation")
        poll_headers = session.get.call_args.kwargs["headers"]
        self.assertEqual(poll_headers["X-ModelScope-Task-Type"], "image_generation")
        self.assertIn("/v1/tasks/task-1", session.get.call_args.args[0])

    async def test_modelscope_submits_reference_images_as_image_urls(self) -> None:
        endpoint = self.endpoint(
            platform="modelscope",
            base_url="https://api-inference.modelscope.cn/",
            model="Qwen/Qwen-Image-2.1",
        )
        session = _session(
            _response(200, {"task_id": "task-1"}),
            gets=(
                _response(
                    200,
                    {"task_status": "SUCCEED", "output_images": [_b64_png_data_url()]},
                ),
            ),
        )
        with patch("aiohttp.ClientSession", return_value=session):
            outcome = await self.run_generation(endpoint, reference=True)

        self.assertTrue(outcome.image_path, outcome.note)
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(len(payload["image_url"]), 1)
        self.assertTrue(payload["image_url"][0].startswith("data:image/png;base64,"))

    async def test_modelscope_reference_block_is_gone(self) -> None:
        # 旧实现会在提交前直接返回“暂不支持参考图输入”，这里确认参考图
        # 请求可以完整走完提交与轮询链路。
        endpoint = self.endpoint(
            platform="modelscope",
            base_url="https://api-inference.modelscope.cn/",
            model="Qwen/Qwen-Image-2.1",
        )
        session = _session(
            _response(200, {"task_id": "task-1"}),
            gets=(
                _response(
                    200,
                    {"task_status": "FAILED", "message": "model busy"},
                ),
            ),
        )
        with patch("aiohttp.ClientSession", return_value=session):
            outcome = await self.run_generation(endpoint, reference=True)

        self.assertFalse(outcome.image_path)
        self.assertIn("model busy", outcome.note)

    # 5. OpenAI 兼容改图 multipart 失败后回退 JSON ---------------

    async def test_openai_compatible_edit_falls_back_to_json_images(self) -> None:
        endpoint = self.endpoint(
            platform="openai",
            base_url="https://proxy.example.test/v1",
            model="sensenova-u1.5-fast",
        )
        session = _session(
            _response(400, {"error": {"message": "invalid arguments"}}),
            _response(200, {"data": [{"b64_json": _b64_png_data_url()}]}),
        )
        with patch("aiohttp.ClientSession", return_value=session):
            outcome = await self.run_generation(endpoint, reference=True)

        self.assertTrue(outcome.image_path, outcome.note)
        self.assertEqual(session.post.call_count, 2)
        first_call = session.post.call_args_list[0]
        second_call = session.post.call_args_list[1]
        self.assertIn("data", first_call.kwargs)  # multipart 表单
        self.assertIn("json", second_call.kwargs)  # JSON images 回退
        json_payload = second_call.kwargs["json"]
        self.assertEqual(json_payload["model"], "sensenova-u1.5-fast")
        self.assertTrue(
            json_payload["images"][0]["image_url"].startswith("data:image/png;base64,")
        )

    async def test_openai_compatible_reference_reaches_edits_endpoint(self) -> None:
        endpoint = self.endpoint(
            platform="openai",
            base_url="https://proxy.example.test/v1",
            model="sensenova-u1.5-fast",
        )
        session = _session(_response(200, {"data": [{"b64_json": _b64_png_data_url()}]}))
        with patch("aiohttp.ClientSession", return_value=session):
            outcome = await self.run_generation(endpoint, reference=True)

        self.assertTrue(outcome.image_path, outcome.note)
        submitted_url = session.post.call_args.args[0]
        self.assertTrue(submitted_url.endswith("/v1/images/edits"), submitted_url)


if __name__ == "__main__":
    unittest.main()
