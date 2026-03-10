#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""双池重试策略模块。

将原 UpstreamClient 中的双池（认证池 + 匿名池）重试决策逻辑提取为
独立的 RetryPolicy 类，以及错误解析工具函数。
所有方法签名与原实现保持一致。
"""

import json
from typing import Any, Dict, Optional, Set, Tuple

from app.core.config import settings
from app.utils.logger import get_logger
from app.utils.token_pool import get_token_pool
from app.utils.guest_session_pool import get_guest_session_pool

logger = get_logger()


# ---------------------------------------------------------------------------
# 错误解析工具
# ---------------------------------------------------------------------------


def extract_upstream_error_details(
    status_code: int,
    error_text: str,
) -> Tuple[Optional[int], str]:
    """解析上游错误响应中的 code/message。

    Args:
        status_code: HTTP 响应状态码。
        error_text: 响应 body 文本。

    Returns:
        ``(error_code, error_message)`` 二元组，解析失败时 code 为 None。
    """
    parsed_code: Optional[int] = None
    parsed_message = (error_text or "").strip()

    try:
        payload = json.loads(error_text)
    except Exception:
        return parsed_code, parsed_message

    if not isinstance(payload, dict):
        return parsed_code, parsed_message

    candidates = [
        payload,
        payload.get("error") if isinstance(payload.get("error"), dict) else None,
        payload.get("detail") if isinstance(payload.get("detail"), dict) else None,
        payload.get("data") if isinstance(payload.get("data"), dict) else None,
    ]

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        code = candidate.get("code")
        if isinstance(code, int):
            parsed_code = code
        elif isinstance(code, str) and code.isdigit():
            parsed_code = int(code)

        for key in ("message", "msg", "detail", "error"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                parsed_message = value.strip()
                break

        if parsed_code is not None or parsed_message:
            break

    return parsed_code, parsed_message


def is_concurrency_limited(
    status_code: int,
    error_code: Optional[int],
    error_message: str,
) -> bool:
    """判断是否为上游并发限制/429 场景。

    Args:
        status_code: HTTP 状态码。
        error_code: 解析出的错误 code（可能为 None）。
        error_message: 解析出的错误消息文本。

    Returns:
        True 表示命中并发限制，需要重试。
    """
    message = (error_message or "").casefold()
    return (
        status_code == 429
        or error_code == 429
        or "concurrency" in message
        or "too many requests" in message
    )


# ---------------------------------------------------------------------------
# 重试策略
# ---------------------------------------------------------------------------


class RetryPolicy:
    """双池重试策略。

    封装认证号池与匿名号池的重试预算计算和切换决策。
    """

    def __init__(self) -> None:
        self.logger = logger

    async def get_guest_retry_limit(self) -> int:
        """匿名号池可提供的最大重试预算。"""
        if not settings.ANONYMOUS_MODE:
            return 0

        guest_pool = get_guest_session_pool()
        if not guest_pool:
            return max(2, settings.GUEST_POOL_SIZE + 1)

        pool_status = guest_pool.get_pool_status()
        available_sessions = int(
            pool_status.get("valid_sessions")
            or pool_status.get("available_sessions")
            or 0
        )
        return max(2, available_sessions + 1)

    async def get_authenticated_retry_limit(self) -> int:
        """认证号池与静态 Token 可提供的最大重试预算。"""
        available_tokens = 0
        token_pool = get_token_pool()
        if token_pool:
            status = await token_pool.get_pool_status()
            available_tokens = int(status.get("available_tokens", 0) or 0)
        return max(0, available_tokens)

    async def get_total_retry_limit(self) -> int:
        """综合认证号池与匿名号池的最大尝试次数。"""
        auth_limit = await self.get_authenticated_retry_limit()
        guest_limit = await self.get_guest_retry_limit()
        return max(1, auth_limit + guest_limit)

    def is_guest_auth(self, transformed: Dict[str, Any]) -> bool:
        """判断当前请求是否使用匿名会话。"""
        return str(transformed.get("auth_mode") or "") == "guest"

    def should_retry_guest_session(
        self,
        status_code: int,
        is_concurrency_limited_flag: bool,
        attempt: int,
        max_attempts: int,
        transformed: Dict[str, Any],
    ) -> bool:
        """判断匿名号池是否需要刷新会话后重试。"""
        return (
            self.is_guest_auth(transformed)
            and (status_code == 401 or is_concurrency_limited_flag)
            and attempt + 1 < max_attempts
        )

    def should_retry_authenticated_session(
        self,
        status_code: int,
        is_concurrency_limited_flag: bool,
        attempt: int,
        max_attempts: int,
        transformed: Dict[str, Any],
    ) -> bool:
        """判断认证号池是否需要切号重试。"""
        current_token = str(transformed.get("token") or "")
        return (
            not self.is_guest_auth(transformed)
            and bool(current_token)
            and (status_code == 401 or is_concurrency_limited_flag)
            and attempt + 1 < max_attempts
        )

    async def release_guest_session(self, transformed: Dict[str, Any]) -> None:
        """释放当前匿名会话占用。"""
        if not self.is_guest_auth(transformed):
            return

        guest_pool = get_guest_session_pool()
        guest_user_id = str(
            transformed.get("guest_user_id") or transformed.get("user_id") or ""
        )
        if guest_pool and guest_user_id:
            guest_pool.release(guest_user_id)

    async def report_guest_session_failure(
        self,
        transformed: Dict[str, Any],
        *,
        is_concurrency_limited_flag: bool = False,
    ) -> None:
        """上报匿名会话失败并补齐新会话。"""
        if not self.is_guest_auth(transformed):
            return

        guest_pool = get_guest_session_pool()
        guest_user_id = str(
            transformed.get("guest_user_id") or transformed.get("user_id") or ""
        )
        if not guest_pool or not guest_user_id:
            return

        if is_concurrency_limited_flag:
            await guest_pool.cleanup_idle_chats()

        await guest_pool.report_failure(guest_user_id)
