"""LLM client for any OpenAI-compatible chat API (default: Gemini's OpenAI endpoint).

Handles: timeouts, retries with exponential backoff + jitter on 429/5xx/network errors,
token-usage accounting, and an optional on-disk response cache (cuts cost of re-running evals).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import openai
from tenacity import (retry, retry_if_exception, stop_after_attempt, wait_random_exponential,
                      before_sleep_log)

from app.config import ROOT_DIR, get_settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM call fails permanently (after retries) or is misconfigured."""


def _daily_quota_exhausted(exc: BaseException) -> bool:
    """A per-day quota 429 only resets at midnight Pacific time, so retrying within seconds is pointless."""
    return isinstance(exc, openai.RateLimitError) and "per day" in str(exc.body).lower()


def _retryable(exc: BaseException) -> bool:
    if _daily_quota_exhausted(exc):
        return False
    if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                        openai.InternalServerError)):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code in (408, 409, 429, 500, 502, 503, 504)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0
    lock: Lock = field(default_factory=Lock, repr=False)

    def add(self, u: Any) -> None:
        with self.lock:
            self.calls += 1
            if u is not None:
                self.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
                self.completion_tokens += getattr(u, "completion_tokens", 0) or 0

    def as_dict(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "calls": self.calls, "cache_hits": self.cache_hits}


class LLMClient:
    def __init__(self, api_key: str, base_url: str, timeout: float, max_retries: int,
                 cache_dir: Path | None = None):
        if not api_key:
            raise LLMError("LLM_API_KEY is not set. Put it in .env (see .env.example).")
        # SDK-level retries disabled: tenacity below gives us backoff+jitter and logging
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.max_retries = max_retries
        self.usage = Usage()
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: Any = None, temperature: float | None = None,
             response_format: dict | None = None, use_cache: bool = True) -> dict:
        """Returns the assistant message as a plain dict: {content, tool_calls: [{id, name, arguments}]}."""
        payload: dict[str, Any] = {"model": model, "messages": messages,
                                   "temperature": get_settings().llm_temperature if temperature is None else temperature}
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format

        key = self._cache_key(payload) if (self.cache_dir and use_cache) else None
        if key and (self.cache_dir / f"{key}.json").exists():
            self.usage.cache_hits += 1
            return json.loads((self.cache_dir / f"{key}.json").read_text(encoding="utf-8"))

        result = self._call(payload)
        if key:
            (self.cache_dir / f"{key}.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    def _call(self, payload: dict) -> dict:
        @retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(self.max_retries + 1),
               wait=wait_random_exponential(multiplier=2, max=40), reraise=True,
               before_sleep=before_sleep_log(log, logging.WARNING))
        def _do():
            return self.client.chat.completions.create(**payload)

        t = time.time()
        try:
            resp = _do()
        except openai.AuthenticationError as e:
            raise LLMError("LLM authentication failed - check LLM_API_KEY") from e
        except openai.BadRequestError as e:
            raise LLMError(f"LLM rejected the request: {str(e.body or e.message)[:400]}") from e
        except openai.RateLimitError as e:
            if _daily_quota_exhausted(e):
                raise LLMError(f"Daily request quota for {payload['model']} is used up (free tier); it resets at "
                               "midnight Pacific time. Set LLM_MODEL to another model or enable billing.") from e
            raise LLMError(f"LLM rate limit still exceeded after retries: {e}") from e
        except Exception as e:
            raise LLMError(f"LLM call failed after retries: {e}") from e
        self.usage.add(resp.usage)
        if not resp.choices:
            raise LLMError("LLM returned no choices (possibly blocked by safety filter)")
        msg = resp.choices[0].message
        log.debug("LLM %s %.1fs", payload["model"], time.time() - t)
        return {
            "content": msg.content or "",
            "finish_reason": resp.choices[0].finish_reason,
            "usage": {"prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                      "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0},
            "tool_calls": [
                # extra_content carries Gemini's thought_signature, which must be echoed back next turn
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}",
                 **({"extra_content": tc.model_extra["extra_content"]} if (tc.model_extra or {}).get("extra_content") else {})}
                for tc in (msg.tool_calls or [])
            ],
        }


def parse_json_loose(text: str) -> Any:
    """Parse JSON that may be wrapped in ```json fences or surrounded by prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for open_c, close_c in (("{", "}"), ("[", "]")):
            s, e = text.find(open_c), text.rfind(close_c)
            if s != -1 and e > s:
                try:
                    return json.loads(text[s:e + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]}")


@lru_cache
def get_llm() -> LLMClient:
    s = get_settings()
    return LLMClient(s.llm_api_key, s.llm_base_url, s.llm_timeout_s, s.llm_max_retries,
                     cache_dir=ROOT_DIR / "data" / "llm_cache")
