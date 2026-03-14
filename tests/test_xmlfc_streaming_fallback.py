#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.response_handler import ResponseHandler
from app.models.schemas import OpenAIRequest


READ_TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
            },
            "required": ["file_path"],
        },
    },
}


class MockResponse:
    def __init__(self, chunks):
        self.chunks = chunks

    async def aiter_lines(self):
        for line in self.chunks:
            yield line


def _build_request() -> OpenAIRequest:
    return OpenAIRequest(
        model="GLM-5-Thinking",
        messages=[{"role": "user", "content": "read files"}],
        tools=[READ_TOOL],
        stream=True,
    )


def _build_transformed():
    return {
        "trigger_signal": "<Function_TEST_Start/>",
        "tools": [READ_TOOL],
        "tool_strategy": "xmlfc",
    }


async def _collect_outputs(chunks):
    handler = ResponseHandler()
    response = MockResponse(chunks)
    outputs = []
    async for item in handler.handle_stream_response(
        response,
        "chat_test",
        "GLM-5-Thinking",
        _build_request(),
        _build_transformed(),
    ):
        outputs.append(item)
    return outputs


def _extract_chunks(outputs):
    payloads = []
    for sse in outputs:
        stripped = sse.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped[5:].strip()
        if payload == "[DONE]":
            continue
        payloads.append(json.loads(payload))
    return payloads


def test_streaming_bare_xml_falls_back_to_tool_calls():
    chunks = [
        'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"继续优化 Dashboard\\n"}}',
        'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"<function_calls>\\n<function_call>\\n<tool>Read</tool>\\n<args_json><![CDATA[{\\"file_path\\": \\"/tmp/a.vue\\"}]]></args_json>\\n</function_call>\\n"}}',
        'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"<function_call>\\n<tool>Read</tool>\\n<args_json><![CDATA[{\\"file_path\\": \\"/tmp/b.vue\\"}]]></args_json>\\n</function_call>\\n</function_calls>"}}',
    ]

    outputs = asyncio.run(_collect_outputs(chunks))
    payloads = _extract_chunks(outputs)

    tool_names = []
    contents = []
    finish_reasons = []
    for payload in payloads:
        choice = payload["choices"][0]
        delta = choice.get("delta", {})
        finish_reasons.append(choice.get("finish_reason"))
        if delta.get("content"):
            contents.append(delta["content"])
        for tool_call in delta.get("tool_calls", []) or []:
            tool_names.append(tool_call["function"]["name"])

    assert tool_names == ["Read", "Read"]
    assert "tool_calls" in finish_reasons
    assert all("<function_calls>" not in content for content in contents)
    assert all("<function_call>" not in content for content in contents)


def test_streaming_bare_xml_with_trailing_text_is_replayed_as_content():
    chunks = [
        'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"这是一个 XML 示例：\\n"}}',
        'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"<function_calls>\\n<function_call>\\n<tool>Read</tool>\\n<args_json><![CDATA[{\\"file_path\\": \\"/tmp/demo.vue\\"}]]></args_json>\\n</function_call>\\n</function_calls>"}}',
        'data: {"type":"chat:completion","data":{"phase":"answer","delta_content":"\\n后面还有解释文本"}}',
        'data: {"type":"chat:completion","data":{"phase":"done","done":true}}',
    ]

    outputs = asyncio.run(_collect_outputs(chunks))
    payloads = _extract_chunks(outputs)

    joined_content = []
    saw_tool_calls = False
    finish_reason = None
    for payload in payloads:
        choice = payload["choices"][0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta", {})
        if delta.get("tool_calls"):
            saw_tool_calls = True
        if delta.get("content"):
            joined_content.append(delta["content"])

    content_text = "".join(joined_content)
    assert saw_tool_calls is False
    assert finish_reason == "stop"
    assert "<function_calls>" in content_text
    assert "后面还有解释文本" in content_text
