#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenAI 消息预处理 + JWT 工具函数。

将原 upstream.py 中的消息规范化函数和 JWT 解析工具提取为独立模块。
所有函数签名和行为与原实现完全一致。
"""

import base64
import json
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# JWT 解析工具
# ---------------------------------------------------------------------------


def _urlsafe_b64decode(data: str) -> bytes:
    """Decode a URL-safe base64 string with proper padding."""
    if isinstance(data, str):
        data_bytes = data.encode("utf-8")
    else:
        data_bytes = data
    padding = b"=" * (-len(data_bytes) % 4)
    return base64.urlsafe_b64decode(data_bytes + padding)


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    """Decode JWT payload without verification to extract metadata."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload_raw = _urlsafe_b64decode(parts[1])
        return json.loads(payload_raw.decode("utf-8", errors="ignore"))
    except Exception:
        return {}


def extract_user_id_from_token(token: str) -> str:
    """Extract user_id from a JWT's payload. Fallback to 'guest'."""
    payload = _decode_jwt_payload(token) if token else {}
    for key in ("id", "user_id", "uid", "sub"):
        val = payload.get(key)
        if isinstance(val, (str, int)) and str(val):
            return str(val)
    return "guest"


# ---------------------------------------------------------------------------
# 消息内容工具
# ---------------------------------------------------------------------------


def _extract_text_from_content(content: Any) -> str:
    """Extract text parts from OpenAI-compatible content payloads."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return " ".join(part for part in parts if part).strip()

    if content is None:
        return ""

    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _stringify_tool_arguments(arguments: Any) -> str:
    """Normalize tool-call arguments into a JSON string."""
    if isinstance(arguments, str):
        return arguments

    try:
        return json.dumps(arguments or {}, ensure_ascii=False)
    except Exception:
        return "{}"


# ---------------------------------------------------------------------------
# 工具调用索引与格式化
# ---------------------------------------------------------------------------


def _build_tool_call_index(
    messages: List[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Index assistant tool calls by id for later tool-result messages."""
    index: Dict[str, Dict[str, str]] = {}

    for message in messages:
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            tool_call_id = tool_call.get("id")
            function_data = (
                tool_call.get("function")
                if isinstance(tool_call.get("function"), dict)
                else {}
            )
            name = str(function_data.get("name", "")).strip()
            if not isinstance(tool_call_id, str) or not name:
                continue

            index[tool_call_id] = {
                "name": name,
                "arguments": _stringify_tool_arguments(
                    function_data.get("arguments")
                ),
            }

    return index


def _format_tool_result_message(
    tool_name: str,
    tool_arguments: str,
    result_content: str,
) -> str:
    """Serialize a tool result into a text block the upstream can consume."""
    return (
        "<tool_execution_result>\n"
        f"<tool_name>{tool_name}</tool_name>\n"
        f"<tool_arguments>{tool_arguments}</tool_arguments>\n"
        f"<tool_output>{result_content}</tool_output>\n"
        "</tool_execution_result>"
    )


def _format_assistant_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    """Serialize historical assistant tool calls into a text block."""
    blocks: List[str] = []

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue

        function_data = (
            tool_call.get("function")
            if isinstance(tool_call.get("function"), dict)
            else {}
        )
        name = str(function_data.get("name", "")).strip()
        if not name:
            continue

        arguments = _stringify_tool_arguments(function_data.get("arguments"))
        blocks.append(
            "<function_call>\n"
            f"<name>{name}</name>\n"
            f"<args_json>{arguments}</args_json>\n"
            "</function_call>"
        )

    if not blocks:
        return ""

    return "<function_calls>\n" + "\n".join(blocks) + "\n</function_calls>"


# ---------------------------------------------------------------------------
# 消息预处理主入口
# ---------------------------------------------------------------------------


def preprocess_openai_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize OpenAI history into shapes accepted by the upstream service.

    处理以下转换：
    - ``developer`` 角色 → ``system``
    - ``tool`` 角色 → ``user``（将工具结果序列化为 XML 文本块）
    - 带有 ``tool_calls`` 的 ``assistant`` → 合并内容与工具调用序列化

    Args:
        messages: OpenAI 格式的消息列表（已 model_dump）。

    Returns:
        上游服务可接受的消息列表。
    """
    tool_call_index = _build_tool_call_index(messages)
    normalized: List[Dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")

        if role == "developer":
            converted = dict(message)
            converted["role"] = "system"
            normalized.append(converted)
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            content = _extract_text_from_content(message.get("content"))
            tool_info = tool_call_index.get(
                tool_call_id,
                {
                    "name": str(message.get("name") or "unknown_tool"),
                    "arguments": "{}",
                },
            )
            normalized.append(
                {
                    "role": "user",
                    "content": _format_tool_result_message(
                        tool_info["name"],
                        tool_info["arguments"],
                        content,
                    ),
                }
            )
            continue

        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            content = _extract_text_from_content(message.get("content"))
            tool_calls_text = _format_assistant_tool_calls(message["tool_calls"])
            merged_content = "\n".join(
                part for part in (content, tool_calls_text) if part
            ).strip()
            normalized.append({"role": "assistant", "content": merged_content})
            continue

        normalized.append(dict(message))

    return normalized


def extract_last_user_text(messages: List[Dict[str, Any]]) -> str:
    """Extract the last user text from the original OpenAI message history."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = _extract_text_from_content(message.get("content"))
        if content:
            return content
    return ""
