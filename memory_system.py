from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

from mem0 import Memory
from mem0.memory.utils import extract_json, remove_code_blocks
from mem0.vector_stores.faiss import FAISS as Mem0FaissStore

from config import AppConfig, require_env
from prompt import (
    USER_MEMORY_EXTRACTION_PROMPT,
    USER_MEMORY_EXTRACTION_WITH_IMPORTANCE_PROMPT,
    IMPORTANCE_RATER_PROMPT,
)


@dataclass
class MemoryAddDebug:
    # 记忆提取LLM（facts extraction）原始输出
    fact_extraction_raw: str = ""
    # 解析后的 facts
    extracted_facts: list[str] = None  # type: ignore[assignment]
    # importance 评分 LLM 原始输出（仅在启用动态重要性时）
    importance_raw: str = ""
    # importance 评分解析结果（仅在启用动态重要性时）
    importance_scored: list[dict[str, Any]] = None  # type: ignore[assignment]
    # mem0.add() 的返回（包含写入/更新/删除的 memories）
    mem0_result: dict[str, Any] | None = None
    error: str | None = None


class MemorySystem:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        if self.cfg.MEM0_VECTOR_STORE_PROVIDER != "faiss":
            raise ValueError(
                f"ONLINE_CHAT only supports faiss for now, got provider={self.cfg.MEM0_VECTOR_STORE_PROVIDER}"
            )

        require_env("OPENAI_API_KEY")

        self._fact_extraction_system_prompt = USER_MEMORY_EXTRACTION_PROMPT

        self.mem0 = Memory.from_config(
            {
                "version": "v1.1",
                "custom_fact_extraction_prompt": self._fact_extraction_system_prompt,
                "vector_store": {
                    "provider": self.cfg.MEM0_VECTOR_STORE_PROVIDER,
                    "config": {
                        "collection_name": self.cfg.COLLECTION_NAME,
                        "path": self.cfg.MEMORY_DB_PATH,
                        "embedding_model_dims": self.cfg.MEM0_EMBEDDING_DIMS,
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "api_key": require_env("OPENAI_API_KEY"),
                        "model": self.cfg.EMBEDDING_MODEL,
                    },
                },
                "llm": {
                    "provider": self.cfg.MEM0_MEMORY_LLM_PROVIDER,
                    "config": {
                        "api_key": require_env("OPENAI_API_KEY"),
                        "model": self.cfg.MEM0_MEMORY_LLM_MODEL,
                    },
                },
            }
        )

        self._last_fact_extraction_raw: str = ""
        self._wrap_mem0_llm_for_capture()

        # 动态重要性和衰减相关状态
        self.add_counter = 0  # 记录 add 调用次数
        self.enable_importance = cfg.ENABLE_DYNAMIC_IMPORTANCE
        self.enable_fast_search = cfg.ENABLE_FAST_SEARCH
        self.last_decay_time: float = time.time()  # 上次衰减时间

        self._backfill_missing_metadata_fields()

    def _wrap_mem0_llm_for_capture(self) -> None:
        orig = self.mem0.llm.generate_response

        def wrapped_generate_response(*, messages, response_format=None, **kwargs):  # type: ignore[no-untyped-def]
            res = orig(messages=messages, response_format=response_format, **kwargs)
            try:
                if (
                    isinstance(messages, list)
                    and len(messages) >= 2
                    and isinstance(messages[0], dict)
                    and messages[0].get("role") == "system"
                ):
                    sys = (messages[0].get("content") or "").strip()
                    if sys == (self._fact_extraction_system_prompt or "").strip(): #只捕获“事实提取”这一跳
                        self._last_fact_extraction_raw = res or ""
            except Exception:
                pass
            return res

        self.mem0.llm.generate_response = wrapped_generate_response  # type: ignore[assignment]

    def _is_euclidean_distance(self) -> bool:
        try:
            vs = self.mem0.vector_store
            if hasattr(vs, "distance_strategy"):
                return str(getattr(vs, "distance_strategy") or "").lower() == "euclidean"
        except Exception:
            pass
        return True

    def _faiss_bulk_update_payloads(self, updates: dict[str, dict[str, Any]]) -> int:
        """批量更新 payload"""
        if not updates:
            return 0
        vs = self.mem0.vector_store

        if isinstance(vs, Mem0FaissStore):
            updated = 0
            with vs._lock:  # type: ignore[attr-defined]
                for vid, payload in updates.items():
                    if vid not in vs.docstore:  # type: ignore[attr-defined]
                        continue
                    vs.docstore[vid] = payload.copy()  # type: ignore[attr-defined]
                    updated += 1
                vs._save()  # type: ignore[attr-defined]
            return updated

        updated = 0
        for vid, payload in updates.items():
            try:
                vs.update(vector_id=vid, vector=None, payload=payload)  # type: ignore[attr-defined]
                updated += 1
            except Exception:
                continue
        return updated

    def _backfill_missing_metadata_fields(self) -> None:
        """
        给旧 memory 补齐缺失字段
        """
        try:
            vs = self.mem0.vector_store
            items = vs.list(filters={"user_id": self.cfg.USER_ID}, limit=200000)  # type: ignore[attr-defined]
        except Exception:
            return

        updates: dict[str, dict[str, Any]] = {}
        for item in items or []:
            try:
                vid = item.id
                payload = dict(item.payload or {})
                changed = False
                if "dynamic_importance" not in payload:
                    payload["dynamic_importance"] = 0.5
                    payload["original_importance"] = 2
                    changed = True
                if "is_expired" not in payload:
                    payload["is_expired"] = False
                    changed = True
                if changed:
                    updates[str(vid)] = payload
            except Exception:
                continue

        if updates:
            self._faiss_bulk_update_payloads(updates)

    def search(
        self, 
        query: str, 
        *, 
        top_k: Optional[int] = None,
        use_fast_search: Optional[bool] = None,
        enable_dynamic_importance: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """搜索记忆"""
        limit = int(top_k if top_k is not None else self.cfg.TOP_K)
        use_fast = use_fast_search if use_fast_search is not None else self.enable_fast_search
        use_importance = enable_dynamic_importance if enable_dynamic_importance is not None else self.enable_importance

        filters = {"is_expired": False} if use_fast else None
        resp = self.mem0.search(query, user_id=self.cfg.USER_ID, limit=limit, filters=filters)
        results = resp.get("results", []) if isinstance(resp, dict) else resp
        results = results or []
        
        out: list[dict[str, Any]] = []
        for m in results:
            metadata = (m.get("metadata") or {}) if isinstance(m, dict) else {}
            
            if "dynamic_importance" not in metadata:
                metadata["dynamic_importance"] = 0.5 #默认
                metadata["original_importance"] = 2
                metadata["is_expired"] = False
            
            memory_entry = {
                "memory": m.get("memory") if isinstance(m, dict) else str(m),
                "score": float(m.get("score") or 0.0) if isinstance(m, dict) else 0.0,
                "metadata": metadata,
                "id": m.get("id", ""), 
            }
            
            # 计算增强分数
            if use_importance:
                original_score = float(memory_entry["score"])
                importance = metadata.get("dynamic_importance", 0.5)
                weight = float(self.cfg.DYNAMIC_IMPORTANCE_WEIGHT)
                
                if self._is_euclidean_distance():
                    # euclidean 距离越小越好
                    enhanced_score = original_score - (weight * importance)
                    memory_entry["enhanced_score"] = enhanced_score
                    memory_entry["original_score"] = original_score
                else:
                    # inner_product
                    enhanced_score = original_score + (weight * importance)
                    memory_entry["enhanced_score"] = enhanced_score
                    memory_entry["original_score"] = original_score
            
            out.append(memory_entry)
        
        # 按融合分数重排序
        if use_importance and out:
            if self._is_euclidean_distance():
                out.sort(key=lambda x: x.get("enhanced_score", x.get("score", 0.0)), reverse=False)
            else:
                out.sort(key=lambda x: x.get("enhanced_score", x.get("score", 0.0)), reverse=True)
        
        return out[:limit]

    def _score_facts_importance(self, facts: list[str]) -> tuple[str, list[dict[str, Any]]]:
        """给 facts 做 0-5 重要性评分 并归一化到 0-1"""
        facts = [(f or "").strip() for f in (facts or []) if (f or "").strip()]
        if not facts:
            return "", []
        
        system_prompt = IMPORTANCE_RATER_PROMPT
        user_prompt = "Input facts (JSON list):\n" + json.dumps(facts, ensure_ascii=False)

        raw = ""
        try:
            raw = self.mem0.llm.generate_response(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"},
            )
        except Exception:
            return "", []

        parsed = self._parse_facts_with_importance(raw or "")
        out: list[dict[str, Any]] = []
        for item in parsed:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            out.append(
                {
                    "content": content,
                    "importance": int(item.get("importance", 0)),
                    "normalized_importance": float(item.get("normalized_importance", 0.0)),
                }
            )
        return raw or "", out

    def add_turn(self, user_text: str, assistant_text: str) -> MemoryAddDebug:
        """写入"用户+助手"这一轮对话到共享记忆库，并返回记忆提取LLM的原始输出"""
        dbg = MemoryAddDebug(extracted_facts=[])
        self._last_fact_extraction_raw = ""
        t = time.time()
        try:
            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
            
            # metadata
            base_metadata = {"timestamp": t}
            
            result = self.mem0.add(messages, user_id=self.cfg.USER_ID, metadata=base_metadata)
            dbg.mem0_result = result if isinstance(result, dict) else {"results": result}

            raw = self._last_fact_extraction_raw or ""
            dbg.fact_extraction_raw = raw
            dbg.extracted_facts = self._parse_facts_from_raw(raw)

            if self.enable_importance:
                imp_raw, scored = self._score_facts_importance(dbg.extracted_facts)
                dbg.importance_raw = imp_raw
                dbg.importance_scored = scored
                self._update_newly_added_memories_metadata(dbg.mem0_result, scored)
            
            # 增加计数器并检查是否需要衰减
            self.add_counter += 1
            if self.add_counter % self.cfg.DECAY_CHECK_INTERVAL == 0:
                self._trigger_decay()
            
            return dbg
        except Exception as e:
            dbg.error = str(e)
            return dbg

    @staticmethod
    def _parse_facts_from_raw(raw: str) -> list[str]:
        try:
            s = remove_code_blocks(raw or "")
            if not s.strip():
                return []
            try:
                facts_data = json.loads(s).get("facts", [])
            except json.JSONDecodeError:
                extracted = extract_json(s)
                facts_data = json.loads(extracted).get("facts", [])
            
            # 兼容两种格式
            result = []
            for item in facts_data:
                if isinstance(item, dict):
                    # 新格式：{"content": "...", "importance": 3}
                    result.append(item.get("content", ""))
                else:
                    # 旧格式：直接是字符串
                    result.append(str(item))
            return result
        except Exception:
            return []

    @staticmethod
    def _parse_facts_with_importance(raw: str) -> list[dict[str, Any]]:
        """
        解析带重要性评分的 facts
        """
        try:
            s = remove_code_blocks(raw or "")
            if not s.strip():
                return []
            try:
                facts_data = json.loads(s).get("facts", [])
            except json.JSONDecodeError:
                extracted = extract_json(s)
                facts_data = json.loads(extracted).get("facts", [])
            
            result = []
            for item in facts_data:
                if isinstance(item, dict) and "content" in item:
                    # 新格式
                    importance = int(item.get("importance", 0))
                    importance = max(0, min(5, importance))  # 限制在 0-5
                    result.append({
                        "content": item.get("content", ""),
                        "importance": importance,
                        "normalized_importance": importance / 5.0,  # 归一化到 0-1
                    })
                else:
                    # 兼容旧格式或纯字符串
                    content = str(item) if not isinstance(item, dict) else item.get("content", str(item))
                    result.append({
                        "content": content,
                        "importance": 2,  # 默认
                        "normalized_importance": 0.4,
                    })
            return result
        except Exception:
            return []

    def debug_state(self) -> dict[str, Any]:
        return {
            "cfg": asdict(self.cfg),
            "collection": self.cfg.COLLECTION_NAME,
            "path": self.cfg.MEMORY_DB_PATH,
            "provider": self.cfg.MEM0_VECTOR_STORE_PROVIDER,
            "add_counter": self.add_counter,
            "next_decay_in": self.cfg.DECAY_CHECK_INTERVAL - (self.add_counter % self.cfg.DECAY_CHECK_INTERVAL),
        }

    def _update_newly_added_memories_metadata(
        self, 
        add_result: Any,
        facts_with_importance: list[dict[str, Any]],
    ) -> None:
        try:
            results = add_result.get("results", []) if isinstance(add_result, dict) else add_result
            results = results or []
        except Exception:
            return

        # content -> importance
        def _norm(s: str) -> str:
            return " ".join((s or "").strip().split()).lower()

        importance_by_content: dict[str, dict[str, Any]] = {}
        for item in facts_with_importance or []:
            c = (item.get("content") or "").strip()
            if not c:
                continue
            importance_by_content[_norm(c)] = item

        updates: dict[str, dict[str, Any]] = {}

        for item in results:
            if not isinstance(item, dict):
                continue
            if str(item.get("event") or "").upper() != "ADD":
                continue
            mid = (item.get("id") or "").strip()
            mem_text = (item.get("memory") or "").strip()
            if not mid:
                continue
            try:
                existing = self.mem0.vector_store.get(vector_id=mid)
                payload = dict(existing.payload or {})
            except Exception:
                payload = {}

            # 默认值
            payload.setdefault("dynamic_importance", 0.5)
            payload.setdefault("original_importance", 2)
            payload.setdefault("is_expired", False)

            matched = importance_by_content.get(_norm(mem_text))
            if matched:
                payload["original_importance"] = int(matched.get("importance", 2))
                payload["dynamic_importance"] = float(matched.get("normalized_importance", 0.4))
                payload["is_expired"] = False

            updates[mid] = payload

        if updates:
            self._faiss_bulk_update_payloads(updates)

    def _trigger_decay(self) -> None:
        """每N次add后触发的全局衰减"""
        try:
            self.last_decay_time = time.time()
            vs = self.mem0.vector_store
            items = vs.list(filters={"user_id": self.cfg.USER_ID, "is_expired": False}, limit=200000)  # type: ignore[attr-defined]
            items = items or []
            updates: dict[str, dict[str, Any]] = {}

            for it in items:
                vid = str(it.id)
                payload = dict(it.payload or {})
                imp = float(payload.get("dynamic_importance", 0.5))
                if imp > 0:
                    imp = (imp * float(self.cfg.DECAY_MULTIPLIER)) + float(self.cfg.DECAY_OFFSET)
                # 低于阈值标记过期
                if imp < float(self.cfg.DECAY_THRESHOLD):
                    payload["is_expired"] = True
                payload["dynamic_importance"] = imp
                payload.setdefault("original_importance", int(round(imp * 5)))
                updates[vid] = payload

            if updates:
                self._faiss_bulk_update_payloads(updates)
        except Exception as e:
            _ = e

    def revive_memories(self, memory_ids: list[str]) -> dict[str, Any]:
        try:
            ids = [str(x) for x in (memory_ids or []) if str(x).strip()]
            if not ids:
                return {"revived_count": 0, "memory_ids": []}

            updates: dict[str, dict[str, Any]] = {}
            for vid in ids:
                try:
                    existing = self.mem0.vector_store.get(vector_id=vid)
                    payload = dict(existing.payload or {})
                except Exception:
                    continue

                imp = float(payload.get("dynamic_importance", 0.5))
                imp = min((imp + float(self.cfg.REVIVE_OFFSET)) * float(self.cfg.REVIVE_MULTIPLIER), float(self.cfg.REVIVE_MAX))
                payload["dynamic_importance"] = imp
                payload["is_expired"] = False
                payload.setdefault("original_importance", int(round(imp * 5)))
                updates[vid] = payload

            updated = self._faiss_bulk_update_payloads(updates)
            return {"revived_count": updated, "memory_ids": ids}
        except Exception as e:
            return {"revived_count": 0, "memory_ids": [], "error": str(e)}

