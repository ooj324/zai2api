import json
from unittest.mock import MagicMock

import pytest

from app.core.response_handler import ResponseHandler, StreamContext
from app.core.toolify.xml_protocol import StreamingFunctionCallDetector
from app.models.schemas import OpenAIRequest


class MockResponse:
    def __init__(self, lines):
        self.lines = lines

    async def aiter_lines(self):
        for line in self.lines:
            yield line


def make_sse_line(delta_content="", phase="answer", done=False):
    payload = {
        "type": "chat:completion",
        "data": {
            "phase": phase,
            "delta_content": delta_content,
        },
    }
    if done:
        payload["data"]["done"] = True
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def make_request(with_tools=False):
    kwargs = {}
    if with_tools:
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "Edit",
                    "parameters": {"type": "object"},
                },
            }
        ]
    return OpenAIRequest(
        model="GLM-5-Thinking",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        **kwargs,
    )


def collect_check_result(
    handler,
    ctx,
    current_text,
    iterations=150,
):
    result = None
    for _ in range(iterations):
        result = handler._check_repetition(ctx, current_text)
        if result:
            break
    return result


async def collect_stream_output(lines, request, transformed=None):
    handler = ResponseHandler()
    handler.logger = MagicMock()
    transformed = transformed or {}
    output = []
    async for chunk in handler.handle_stream_response(
        MockResponse(lines),
        "test-chat",
        request.model,
        request,
        transformed,
    ):
        output.append(chunk)
    return output


def parse_sse_chunk(raw):
    if raw.strip() == "data: [DONE]":
        return "[DONE]"
    assert raw.startswith("data: ")
    return json.loads(raw[len("data: ") :])


def has_repetition_error(output):
    for raw in output:
        if raw.strip() == "data: [DONE]":
            continue
        payload = parse_sse_chunk(raw)
        delta = payload["choices"][0]["delta"]
        if "repetition loop detected" in delta.get("content", ""):
            return True
    return False


def has_tool_calls(output):
    for raw in output:
        if raw.strip() == "data: [DONE]":
            continue
        payload = parse_sse_chunk(raw)
        delta = payload["choices"][0]["delta"]
        if delta.get("tool_calls"):
            return True
    return False


def test_detect_repetition_loop_detects_simple_cycle():
    pattern = ResponseHandler._detect_repetition_loop("abc123XYZ_" * 12)
    assert pattern == "abc123XYZ_"


def test_detect_repetition_loop_ignores_repeated_markup_suffix_without_true_cycle():
    pattern = ResponseHandler._detect_repetition_loop(
        "".join(
            f"field_{index}></label>\n          <label>\n"
            for index in range(8)
        )
    )
    assert pattern is None


@pytest.mark.parametrize(
    "text",
    [
        "ASCII start 中文段落 😀 emoji line\nSecond line with tab\tand entity "
        "&lt;ok&gt;\\u4f60\\u597d end",
        "line1\r\nline2\r\nline3\r\nline4\r\nline5\r\nline6\r\nline7\r\n"
        "line8\r\nline9\r\nline10\r\n",
        r'{"a":"\u4f60\u597d","b":"\u4e16\u754c","c":"\u6d4b\u8bd5"}',
    ],
)
def test_detect_repetition_loop_ignores_mixed_nonrepetitive_encoding_samples(text):
    pattern = ResponseHandler._detect_repetition_loop(text)
    assert pattern is None


def test_check_repetition_skips_thinking_phase():
    handler = ResponseHandler()
    ctx = StreamContext(chat_id="test", model="glm-5")
    ctx.last_phase = "thinking"

    result = collect_check_result(handler, ctx, "abc123XYZ_")

    assert result is None
    assert ctx.repeat_chunk_count == 0
    assert ctx.repeat_buffer == ""


def test_check_repetition_detects_obvious_loop_after_threshold():
    handler = ResponseHandler()
    ctx = StreamContext(chat_id="test", model="glm-5")
    ctx.last_phase = "answer"

    result = collect_check_result(handler, ctx, "abc123XYZ_")

    assert result is not None
    assert any("repetition loop detected" in chunk for chunk in result)


def test_check_repetition_should_ignore_tool_parsing_chunks():
    handler = ResponseHandler()
    ctx = StreamContext(chat_id="test", model="glm-5")
    ctx.last_phase = "answer"
    ctx.detector = StreamingFunctionCallDetector("<Function_TEST_Start/>")
    ctx.detector.state = "tool_parsing"

    result = collect_check_result(handler, ctx, "abc123XYZ_")

    assert result is None
    assert ctx.repeat_chunk_count == 0
    assert ctx.repeat_buffer == ""


@pytest.mark.asyncio
async def test_handle_stream_response_detects_real_repetition_loop():
    output = await collect_stream_output(
        [make_sse_line("abc123XYZ_") for _ in range(120)],
        make_request(),
    )

    assert has_repetition_error(output) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "chunk"),
    [
        ("chinese", "你好世界"),
        ("emoji", "😀🚀✨"),
        ("unicode_escape", r"\u4f60\u597d\u4e16\u754c"),
        ("crlf", "line1\r\n"),
        ("tab", "\t<item>\n"),
        ("xml_entity", "&lt;div&gt;hello&lt;/div&gt;"),
    ],
)
async def test_handle_stream_response_detects_encoded_repetition_loops(label, chunk):
    output = await collect_stream_output(
        [make_sse_line(chunk) for _ in range(120)],
        make_request(),
    )

    assert has_repetition_error(output) is True, label


@pytest.mark.asyncio
async def test_handle_stream_response_ignores_mixed_multilingual_nonrepetitive_stream():
    lines = [
        make_sse_line(
            f"chunk-{index}: 中文 {index} 😀 line\\n"
            f"tab\\tvalue-{index} entity &lt;tag-{index}&gt; "
            f"escape \\\\u4f60\\\\u597d-{index}"
        )
        for index in range(110)
    ]
    lines[-1] = make_sse_line(
        "final-chunk: 中文 done 😀 line\\n"
        "tab\\tvalue-done entity &lt;tag-done&gt; escape \\\\u4f60\\\\u597d-done",
        done=True,
    )

    output = await collect_stream_output(lines, make_request())

    assert has_repetition_error(output) is False


@pytest.mark.asyncio
async def test_handle_stream_response_parses_short_tool_xml():
    lines = [
        make_sse_line("准备调用工具。\n"),
        make_sse_line("<Function_TEST_Start/>\n<function_calls>\n<"),
        make_sse_line("function_call>\n<tool>Edit</tool>\n<args_"),
        make_sse_line(
            'json><![CDATA[{"file_path":"/tmp/a.txt","new_string":"ok"}]]>'
            "</args_json>\n</function_call>\n</function_calls>",
            done=True,
        ),
    ]
    output = await collect_stream_output(
        lines,
        make_request(with_tools=True),
        transformed={
            "trigger_signal": "<Function_TEST_Start/>",
            "tool_strategy": "xmlfc",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "Edit",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert has_repetition_error(output) is False
    assert has_tool_calls(output) is True


@pytest.mark.asyncio
async def test_handle_stream_response_should_not_abort_on_large_edit_tool_xml():
    lines = [
        make_sse_line("准备编辑页面。\n"),
        make_sse_line(
            "<Function_TEST_Start/>\n"
            "<function_calls>\n"
            "<function_call>\n"
            "<tool>Edit</tool>\n"
            '<args_json><![CDATA[{"file_path":"/tmp/TenantsPage.vue","old_string":"'
        ),
    ]
    lines.extend(make_sse_line(">\\n       ") for _ in range(130))
    lines.append(
        make_sse_line(
            '","new_string":"ok"}]]></args_json>\n'
            "</function_call>\n"
            "</function_calls>",
            done=True,
        )
    )
    output = await collect_stream_output(
        lines,
        make_request(with_tools=True),
        transformed={
            "trigger_signal": "<Function_TEST_Start/>",
            "tool_strategy": "xmlfc",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "Edit",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert has_repetition_error(output) is False
    assert has_tool_calls(output) is True
