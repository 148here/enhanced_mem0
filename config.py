from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _load_root_env() -> None:
    """
    读取“仓库根目录（当前工作目录）”下的 `.env`。
    - 用户要求：直接读取当前目录下的 env
    - evaluation 也使用 dotenv.load_dotenv() 读取 .env
    """
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)


_load_root_env()


@dataclass(frozen=True)
class AppConfig:
    # =========================
    # 单用户（默认只有一个用户）
    # =========================
    USER_ID: str = os.getenv("USER_ID", "default_user")

    # =========================
    # 与 evaluation 保持一致的 env 理解
    # =========================
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL: str = os.getenv("MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    MEM0_VECTOR_STORE_PROVIDER: str = os.getenv("MEM0_VECTOR_STORE_PROVIDER", "faiss")
    MEM0_MEMORY_LLM_PROVIDER: str = os.getenv("MEM0_MEMORY_LLM_PROVIDER", "openai")
    MEM0_MEMORY_LLM_MODEL: str = os.getenv("MEM0_MEMORY_LLM_MODEL", os.getenv("MODEL", "gpt-4o-mini"))
    MEM0_EMBEDDING_DIMS: int = int(os.getenv("MEM0_EMBEDDING_DIMS", "1536"))

    # =========================
    # 在线对话系统自身配置（可覆盖 env）
    # =========================
    MEMORY_DB_PATH: str = os.getenv("MEM0_VECTOR_STORE_PATH", "chat_memory/faiss")
    COLLECTION_NAME: str = os.getenv("MEM0_COLLECTION", "online_chat")

    # 记忆检索数量（top_k）
    TOP_K: int = int(os.getenv("ONLINE_CHAT_TOP_K", "10"))

    # 对话历史最大可见范围：最近 N 轮（1轮=用户+助手）
    MAX_HISTORY_TURNS: int = int(os.getenv("ONLINE_CHAT_MAX_HISTORY_TURNS", "3"))

    # WebUI
    WEBUI_HOST: str = os.getenv("ONLINE_CHAT_WEBUI_HOST", "127.0.0.1")
    WEBUI_PORT: int = int(os.getenv("ONLINE_CHAT_WEBUI_PORT", "7860"))

    # =========================
    # 裁判模型+动态topk配置（默认关闭，保持向后兼容）
    # =========================
    ENABLE_JUDGE_AND_DYNAMIC_TOPK: bool = os.getenv("ONLINE_CHAT_ENABLE_JUDGE", "false").lower() == "true"
    NUM_CANDIDATES: int = int(os.getenv("ONLINE_CHAT_NUM_CANDIDATES", "3"))
    MAX_EXPAND_ROUNDS: int = int(os.getenv("ONLINE_CHAT_MAX_EXPAND_ROUNDS", "2"))
    EXPAND_STEP: int = int(os.getenv("ONLINE_CHAT_EXPAND_STEP", "4"))
    CANDIDATE_TEMPERATURE: float = float(os.getenv("ONLINE_CHAT_CANDIDATE_TEMPERATURE", "0.2"))

    # =========================
    # 动态重要性配置
    # =========================
    ENABLE_DYNAMIC_IMPORTANCE: bool = os.getenv("ONLINE_CHAT_ENABLE_DYNAMIC_IMPORTANCE", "false").lower() == "true"
    DYNAMIC_IMPORTANCE_WEIGHT: float = float(os.getenv("ONLINE_CHAT_DYNAMIC_IMPORTANCE_WEIGHT", "0.1"))

    # =========================
    # 记忆衰减配置
    # =========================
    DECAY_CHECK_INTERVAL: int = int(os.getenv("ONLINE_CHAT_DECAY_CHECK_INTERVAL", "5"))  # 每N次add触发衰减
    DECAY_MULTIPLIER: float = float(os.getenv("ONLINE_CHAT_DECAY_MULTIPLIER", "0.99"))
    DECAY_OFFSET: float = float(os.getenv("ONLINE_CHAT_DECAY_OFFSET", "-0.002"))
    DECAY_THRESHOLD: float = float(os.getenv("ONLINE_CHAT_DECAY_THRESHOLD", "-0.5"))

    # =========================
    # 记忆复活配置
    # =========================
    REVIVE_MULTIPLIER: float = float(os.getenv("ONLINE_CHAT_REVIVE_MULTIPLIER", "1.01"))
    REVIVE_OFFSET: float = float(os.getenv("ONLINE_CHAT_REVIVE_OFFSET", "0.002"))
    REVIVE_MAX: float = float(os.getenv("ONLINE_CHAT_REVIVE_MAX", "1.0"))

    # =========================
    # 快速搜索配置
    # =========================
    ENABLE_FAST_SEARCH: bool = os.getenv("ONLINE_CHAT_ENABLE_FAST_SEARCH", "false").lower() == "true"


def require_env(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise ValueError(f"Missing required environment variable: {key}")
    return v

