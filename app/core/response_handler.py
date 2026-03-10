#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""流式和非流式响应处理模块。

将原 UpstreamClient._handle_stream_response()、_handle_non_stream_response()
及辅助方法提取为 ResponseHandler 类。
所有参数和行为与原实现完全一致。
"""

import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.core.config import settings
from app.core.openai_compat import (
    create_openai_chunk,
    create_openai_response_with_reasoning,
    format_sse_chunk,
    handle_error,
)
from app.models.schemas import OpenAIRequest
from app.utils.logger import get_logger
from app.utils.tool_call_handler import parse_and_extract_tool_calls

logger = get_logger()


class ResponseHandler:
    """上游 SSE 响应转换为 OpenAI 格式的处理器。

    提供流式和非流式两种处理路径，以及辅助的内容清理方法。
    """

    def __init__(self) -> None:
        self.logger = logger

    # ------------------------------------------------------------------
    # 内容辅助方法
    # ------------------------------------------------------------------

    def clean_reasoning_delta(self, delta_content: str) -> str:
        """清理思考阶段的 details 包裹内容。"""
        if not delta_content:
            return ""

        if delta_content.startswith("<details"):
            if "</summary>\n>" in delta_content:
                return delta_content.split("</summary>\n>")[-1].strip()
            if "</summary>\n" in delta_content:
                return delta_content.split("</summary>\n")[-1].lstrip("> ").strip()

        return delta_content

    def extract_answer_content(self, text: str) -> str:
        """提取思考结束后的答案正文。"""
        if not text:
            return ""

        if "</details>\n" in text:
            return text.split("</details>\n")[-1]

        if "</details>" in text:
            return text.split("</details>")[-1].lstrip()

        return text

    def normalize_tool_calls(
        self,
        raw_tool_calls: Any,
        start_index: int = 0,
    ) -> List[Dict[str, Any]]:
        """标准化上游工具调用为 OpenAI 兼容格式。"""
        if not raw_tool_calls:
            return []

        tool_calls = (
            raw_tool_calls if isinstance(raw_tool_calls, list) else [raw_tool_calls]
        )
        normalized: List[Dict[str, Any]] = []

        for offset, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue

            function_data = tool_call.get("function") or {}
            normalized.append(
                {
                    "index": tool_call.get("index", start_index + offset),
                    "id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": function_data.get("name", ""),
                        "arguments": function_data.get("arguments", ""),
                    },
                }
            )

        return normalized

    def format_search_results(self, data: Dict[str, Any]) -> str:
        """将上游搜索结果格式化为可追加的 Markdown 引用。"""
        search_info = (
            data.get("results") or data.get("sources") or data.get("citations")
        )
        if not isinstance(search_info, list) or not search_info:
            return ""

        citations = []
        for index, item in enumerate(search_info, 1):
            if not isinstance(item, dict):
                continue

            title = item.get("title") or item.get("name") or f"Result {index}"
            url = item.get("url") or item.get("link")
            if url:
                citations.append(f"[{index}] [{title}]({url})")

        if not citations:
            return ""

        return "\n\n---\n" + "\n".join(citations)

    # ------------------------------------------------------------------
    # 流式响应处理
    # ------------------------------------------------------------------

    async def handle_stream_response(
        self,
        response: httpx.Response,
        chat_id: str,
        model: str,
        request: OpenAIRequest,
        transformed: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """处理上游流式响应，转换为 OpenAI SSE 流。"""
        self.logger.info("✅ 上游响应成功，开始处理 SSE 流")

        has_tools = settings.TOOL_SUPPORT and bool(request.tools)
        buffered_content = ""
        usage_info: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        tool_calls_accum: List[Dict[str, Any]] = []
        has_sent_role = False
        finished = False
        line_count = 0

        async def ensure_role_sent() -> Optional[str]:
            nonlocal has_sent_role
            if has_sent_role:
                return None
            has_sent_role = True
            return format_sse_chunk(
                create_openai_chunk(chat_id, model, {"role": "assistant"})
            )

        async def finalize_stream() -> AsyncGenerator[str, None]:
            nonlocal finished, tool_calls_accum
            if finished:
                return

            if has_tools and not tool_calls_accum:
                parsed_tool_calls, _ = parse_and_extract_tool_calls(buffered_content)
                normalized = self.normalize_tool_calls(parsed_tool_calls)
                if normalized:
                    tool_calls_accum = normalized
                    role_output = await ensure_role_sent()
                    if role_output:
                        yield role_output
                    for tool_call in normalized:
                        yield format_sse_chunk(
                            create_openai_chunk(
                                chat_id, model, {"tool_calls": [tool_call]}
                            )
                        )

            if not has_sent_role:
                role_output = await ensure_role_sent()
                if role_output:
                    yield role_output

            finish_reason = "tool_calls" if tool_calls_accum else "stop"
            finish_chunk = create_openai_chunk(chat_id, model, {}, finish_reason)
            finish_chunk["usage"] = usage_info
            yield format_sse_chunk(finish_chunk)
            yield "data: [DONE]\n\n"
            finished = True

        try:
            async for line in response.aiter_lines():
                line_count += 1
                if not line:
                    continue

                # 调试：捕获原始 SSE 行
                if line_count <= 3:
                    self.logger.info(f"🔍 SSE line #{line_count}: {line[:200]}")

                current_line = line.strip()
                if not current_line.startswith("data:"):
                    if line_count <= 3:
                        self.logger.info(
                            f"🔍 SSE line #{line_count} 跳过 (non-data): "
                            f"{current_line[:100]}"
                        )
                    continue

                chunk_str = current_line[5:].strip()
                if not chunk_str:
                    continue

                if chunk_str == "[DONE]":
                    async for final_chunk in finalize_stream():
                        yield final_chunk
                    continue

                try:
                    chunk = json.loads(chunk_str)
                except json.JSONDecodeError as error:
                    self.logger.debug(
                        f"❌ JSON解析错误: {error}, 内容: {chunk_str[:1000]}"
                    )
                    continue

                # 调试：捕获前3个 SSE chunk 的原始结构
                if line_count <= 3:
                    self.logger.info(
                        f"🔍 SSE raw #{line_count}: type={chunk.get('type')}, "
                        f"keys={list(chunk.keys())}, "
                        f"data_keys={list(chunk.get('data', {}).keys()) if isinstance(chunk.get('data'), dict) else 'N/A'}, "
                        f"raw={json.dumps(chunk, ensure_ascii=False)[:300]}"
                    )

                chunk_type = chunk.get("type")
                data = (
                    chunk.get("data", {}) if chunk_type == "chat:completion" else chunk
                )
                if not isinstance(data, dict):
                    continue

                phase = data.get("phase")
                delta_content = data.get("delta_content", "")
                edit_content = data.get("edit_content", "")

                # 检测上游 SSE 事件中的错误
                sse_error = data.get("error")
                if isinstance(sse_error, dict):
                    error_code_val = sse_error.get("code", "UNKNOWN")
                    error_detail = sse_error.get("detail", "Unknown upstream error")
                    self.logger.error(
                        f"❌ 上游 SSE 返回错误: code={error_code_val}, detail={error_detail}"
                    )
                    error_response = {
                        "error": {
                            "message": error_detail,
                            "type": "upstream_error",
                            "code": error_code_val,
                        }
                    }
                    role_output = await ensure_role_sent()
                    if role_output:
                        yield role_output
                    yield f"data: {json.dumps(error_response)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                if phase and phase != getattr(self, "_last_phase", None):
                    self.logger.info(f"📈 SSE 阶段: {phase}")
                    self._last_phase = phase

                if data.get("usage"):
                    usage_info = data["usage"]

                if delta_content:
                    buffered_content += delta_content
                elif edit_content:
                    buffered_content += edit_content

                direct_tool_calls = self.normalize_tool_calls(
                    data.get("tool_calls"),
                    len(tool_calls_accum),
                )
                if direct_tool_calls:
                    role_output = await ensure_role_sent()
                    if role_output:
                        yield role_output
                    tool_calls_accum.extend(direct_tool_calls)
                    for tool_call in direct_tool_calls:
                        yield format_sse_chunk(
                            create_openai_chunk(
                                chat_id, model, {"tool_calls": [tool_call]}
                            )
                        )

                if phase == "thinking" and delta_content:
                    cleaned = self.clean_reasoning_delta(delta_content)
                    if cleaned:
                        role_output = await ensure_role_sent()
                        if role_output:
                            yield role_output
                        yield format_sse_chunk(
                            create_openai_chunk(
                                chat_id, model, {"reasoning_content": cleaned}
                            )
                        )

                elif phase == "answer":
                    text = delta_content or self.extract_answer_content(edit_content)
                    if text:
                        role_output = await ensure_role_sent()
                        if role_output:
                            yield role_output
                        yield format_sse_chunk(
                            create_openai_chunk(chat_id, model, {"content": text})
                        )

                elif phase == "other":
                    other_text = self.extract_answer_content(edit_content)
                    if other_text:
                        role_output = await ensure_role_sent()
                        if role_output:
                            yield role_output
                        yield format_sse_chunk(
                            create_openai_chunk(
                                chat_id, model, {"content": other_text}
                            )
                        )

                elif phase == "search" or chunk_type == "web_search":
                    citation_text = self.format_search_results(data)
                    if citation_text:
                        role_output = await ensure_role_sent()
                        if role_output:
                            yield role_output
                        yield format_sse_chunk(
                            create_openai_chunk(
                                chat_id, model, {"content": citation_text}
                            )
                        )

                if data.get("done"):
                    async for final_chunk in finalize_stream():
                        yield final_chunk
                    return

            self.logger.info(
                f"✅ SSE 流处理完成，共处理 {line_count} 行数据"
            )

            if not finished:
                async for final_chunk in finalize_stream():
                    yield final_chunk

        except Exception as e:
            self.logger.error(f"❌ 流式响应处理错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            yield format_sse_chunk(
                create_openai_chunk(chat_id, model, {}, "stop")
            )
            yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # 非流式响应处理
    # ------------------------------------------------------------------

    async def handle_non_stream_response(
        self,
        response: httpx.Response,
        chat_id: str,
        model: str,
    ) -> Dict[str, Any]:
        """处理非流式响应，聚合上游 SSE 为一次性 OpenAI 响应。"""
        final_content = ""
        reasoning_content = ""
        tool_calls_accum: List[Dict[str, Any]] = []
        usage_info: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
            async for line in response.aiter_lines():
                if not line:
                    continue

                line = line.strip()
                if not line.startswith("data:"):
                    try:
                        maybe_err = json.loads(line)
                        if isinstance(maybe_err, dict) and (
                            "error" in maybe_err
                            or "code" in maybe_err
                            or "message" in maybe_err
                        ):
                            msg = (
                                (maybe_err.get("error") or {}).get("message")
                                if isinstance(maybe_err.get("error"), dict)
                                else maybe_err.get("message")
                            ) or "上游返回错误"
                            return handle_error(Exception(msg), "API响应")
                    except Exception:
                        pass
                    continue

                data_str = line[5:].strip()
                if not data_str or data_str in ("[DONE]", "DONE", "done"):
                    continue

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                chunk_type = chunk.get("type")
                data = (
                    chunk.get("data", {}) if chunk_type == "chat:completion" else chunk
                )
                if not isinstance(data, dict):
                    continue

                phase = data.get("phase")
                delta_content = data.get("delta_content", "")
                edit_content = data.get("edit_content", "")

                if data.get("usage"):
                    usage_info = data["usage"]

                if phase == "thinking" and delta_content:
                    reasoning_content += self.clean_reasoning_delta(delta_content)

                elif phase == "answer":
                    if delta_content:
                        final_content += delta_content
                    elif edit_content:
                        final_content += self.extract_answer_content(edit_content)

                elif phase == "other" and edit_content:
                    final_content += self.extract_answer_content(edit_content)

                elif phase == "search" or chunk_type == "web_search":
                    final_content += self.format_search_results(data)

                tool_calls_accum.extend(
                    self.normalize_tool_calls(
                        data.get("tool_calls"),
                        len(tool_calls_accum),
                    )
                )

        except Exception as e:
            self.logger.error(f"❌ 非流式响应处理错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return handle_error(e, "非流式聚合")

        if not tool_calls_accum:
            parsed_tool_calls, cleaned_content = parse_and_extract_tool_calls(
                final_content
            )
            normalized = self.normalize_tool_calls(parsed_tool_calls)
            if normalized:
                tool_calls_accum = normalized
                final_content = cleaned_content

        final_content = (final_content or "").strip()
        reasoning_content = (reasoning_content or "").strip()

        if not final_content and reasoning_content:
            final_content = reasoning_content

        return create_openai_response_with_reasoning(
            chat_id,
            model,
            final_content,
            reasoning_content,
            usage_info,
            tool_calls_accum or None,
        )
