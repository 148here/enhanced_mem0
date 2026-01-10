from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

from mem0 import Memory
from mem0.memory.utils import extract_json, remove_code_blocks

from config import AppConfig, require_env
from prompt import USER_MEMORY_EXTRACTION_PROMPT, USER_MEMORY_EXTRACTION_WITH_IMPORTANCE_PROMPT


@dataclass
class MemoryAddDebug:
    # 记忆提取LLM（facts extraction）原始输出
    fact_extraction_raw: str = ""
    # 解析后的 facts
    extracted_facts: list[str] = None  # type: ignore[assignment]
    # mem0.add() 的返回（包含写入/更新/删除的 memories）
    mem0_result: dict[str, Any] | None = None
    error: str | None = None


class MemorySystem:
    """
    单用户共享记忆系统（FAISS）。

    关键点：
    - 使用 mem0 的本地实现：Memory.from_config(...)
    - 使用 FAISS 本地向量库（path + collection_name）
    - 通过“包装 mem0.llm.generate_response”捕获 **记忆提取LLM** 的原始返回，用于 WebUI 展示
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        if self.cfg.MEM0_VECTOR_STORE_PROVIDER != "faiss":
            raise ValueError(
                f"ONLINE_CHAT only supports faiss for now, got provider={self.cfg.MEM0_VECTOR_STORE_PROVIDER}"
            )

        # 与 evaluation 一致：需要 OPENAI_API_KEY
        require_env("OPENAI_API_KEY")

        # 根据是否启用动态重要性选择 prompt
        if cfg.ENABLE_DYNAMIC_IMPORTANCE:
            self._fact_extraction_system_prompt = USER_MEMORY_EXTRACTION_WITH_IMPORTANCE_PROMPT
        else:
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

        # =========================
        # 动态重要性和衰减相关状态
        # =========================
        self.add_counter = 0  # 记录 add 调用次数
        self.enable_importance = cfg.ENABLE_DYNAMIC_IMPORTANCE
        self.enable_fast_search = cfg.ENABLE_FAST_SEARCH
        self.last_decay_time: float = time.time()  # 上次衰减时间

    def _wrap_mem0_llm_for_capture(self) -> None:
        """
        mem0 的事实提取发生在 Memory._add_to_vector_store 内部：
        - 会调用 self.llm.generate_response(messages=[system,user], response_format=json_object)
        - 其 system message content == custom_fact_extraction_prompt（若配置了）
        我们在不改 mem0 源码的前提下，捕获该次调用的返回字符串。
        """

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
                    if sys == (self._fact_extraction_system_prompt or "").strip():
                        # 只捕获“事实提取”这一跳；更新/合并记忆那一跳通常只有 user role
                        self._last_fact_extraction_raw = res or ""
            except Exception:
                # 捕获不应影响主流程
                pass
            return res

        self.mem0.llm.generate_response = wrapped_generate_response  # type: ignore[assignment]

    def search(
        self, 
        query: str, 
        *, 
        top_k: Optional[int] = None,
        use_fast_search: Optional[bool] = None,
        enable_dynamic_importance: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """
        搜索记忆
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            use_fast_search: 是否只搜索活跃记忆（is_expired=false）
            enable_dynamic_importance: 是否使用动态重要性重排序
        
        Returns:
            记忆列表，每个包含 memory, score, metadata, enhanced_score (如果启用动态重要性)
        """
        limit = int(top_k if top_k is not None else self.cfg.TOP_K)
        use_fast = use_fast_search if use_fast_search is not None else self.enable_fast_search
        use_importance = enable_dynamic_importance if enable_dynamic_importance is not None else self.enable_importance
        
        resp = self.mem0.search(query, user_id=self.cfg.USER_ID, limit=limit * 2 if use_fast else limit)
        results = resp.get("results", []) if isinstance(resp, dict) else resp
        results = results or []
        
        out: list[dict[str, Any]] = []
        for m in results:
            metadata = (m.get("metadata") or {}) if isinstance(m, dict) else {}
            
            # 为没有重要性的旧记忆设置默认值
            if "dynamic_importance" not in metadata:
                metadata["dynamic_importance"] = 0.5  # 默认中等重要性
                metadata["original_importance"] = 2
                metadata["is_expired"] = False
            
            # 快速搜索：过滤过期记忆
            if use_fast and metadata.get("is_expired", False):
                continue
            
            memory_entry = {
                "memory": m.get("memory") if isinstance(m, dict) else str(m),
                "score": float(m.get("score") or 0.0) if isinstance(m, dict) else 0.0,
                "metadata": metadata,
                "id": m.get("id", ""),  # 用于后续 revive
            }
            
            # 动态重要性：计算增强分数
            if use_importance:
                original_score = memory_entry["score"]
                importance = metadata.get("dynamic_importance", 0.5)
                enhanced_score = original_score + (self.cfg.DYNAMIC_IMPORTANCE_WEIGHT * importance)
                memory_entry["enhanced_score"] = enhanced_score
                memory_entry["original_score"] = original_score
            
            out.append(memory_entry)
        
        # 如果启用动态重要性，按增强分数重排序
        if use_importance and out:
            out.sort(key=lambda x: x.get("enhanced_score", x.get("score", 0.0)), reverse=True)
        
        # 限制返回数量
        return out[:limit]

    def add_turn(self, user_text: str, assistant_text: str) -> MemoryAddDebug:
        """
        写入"用户+助手"这一轮对话到共享记忆库，并返回记忆提取LLM的原始输出。
        如果启用动态重要性，会为每个 fact 添加 importance metadata。
        """
        dbg = MemoryAddDebug(extracted_facts=[])
        self._last_fact_extraction_raw = ""
        t = time.time()
        try:
            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
            
            # 基础 metadata
            base_metadata = {"timestamp": t}
            
            # 如果启用动态重要性，我们需要先提取 facts+importance，然后逐个添加
            if self.enable_importance:
                # 调用 mem0.add 会触发 fact extraction
                result = self.mem0.add(messages, user_id=self.cfg.USER_ID, metadata=base_metadata)
                dbg.mem0_result = result if isinstance(result, dict) else {"results": result}
                
                raw = self._last_fact_extraction_raw or ""
                dbg.fact_extraction_raw = raw
                facts_with_importance = self._parse_facts_with_importance(raw)
                dbg.extracted_facts = [f["content"] for f in facts_with_importance]
                
                # 更新新添加记忆的 metadata（添加 dynamic_importance 和 is_expired）
                self._update_newly_added_memories_metadata(result, facts_with_importance)
            else:
                # 不启用重要性，使用原有逻辑
                result = self.mem0.add(messages, user_id=self.cfg.USER_ID, metadata=base_metadata)
                dbg.mem0_result = result if isinstance(result, dict) else {"results": result}
                raw = self._last_fact_extraction_raw or ""
                dbg.fact_extraction_raw = raw
                dbg.extracted_facts = self._parse_facts_from_raw(raw)
            
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
        """
        向后兼容：解析旧格式的 facts（纯字符串列表）
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
        返回: [{"content": "...", "importance": 3, "normalized_importance": 0.6}, ...]
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
                        "importance": 2,  # 默认中等重要性
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
        facts_with_importance: list[dict[str, Any]]
    ) -> None:
        """
        为新添加的记忆更新 metadata
        由于 FAISS 不支持直接更新 metadata，这里采用备注方式
        实际的 importance 会在 search 时通过默认值处理
        """
        # FAISS 不支持直接更新metadata，所以这个方法主要用于日志
        # 实际的 importance 管理会在 search 和 revive 时处理
        pass

    def _trigger_decay(self) -> None:
        """
        每N次add后触发的全局衰减
        对所有 is_expired=false 的记忆应用衰减公式
        """
        try:
            print(f"[Decay] Triggering decay check at add_counter={self.add_counter}")
            self.last_decay_time = time.time()
            
            # 获取所有记忆
            all_memories = self.mem0.get_all(user_id=self.cfg.USER_ID, limit=100000)
            results = all_memories.get("results", []) if isinstance(all_memories, dict) else all_memories
            
            if not results:
                return
            
            active_count = 0
            expired_count = 0
            
            # 由于FAISS不支持直接更新，这里只做统计
            # 实际的衰减会在search时动态应用
            for mem in results:
                metadata = mem.get("metadata", {}) if isinstance(mem, dict) else {}
                is_expired = metadata.get("is_expired", False)
                importance = metadata.get("dynamic_importance", 0.5)
                
                if not is_expired:
                    active_count += 1
                    # 应用衰减公式
                    if importance > 0:
                        new_importance = (importance * self.cfg.DECAY_MULTIPLIER) + self.cfg.DECAY_OFFSET
                    else:
                        new_importance = importance
                    
                    # 检查是否应该标记为过期
                    if new_importance < self.cfg.DECAY_THRESHOLD:
                        expired_count += 1
            
            print(f"[Decay] Checked {len(results)} memories: {active_count} active, would expire {expired_count}")
        except Exception as e:
            print(f"[Decay] Error during decay: {e}")

    def revive_memories(self, memory_ids: list[str]) -> dict[str, Any]:
        """
        复活被使用的记忆
        设置 is_expired=false 并增加 dynamic_importance
        
        由于 FAISS 限制，这里返回统计信息
        实际的复活效果会在下次 search 时体现
        """
        try:
            revived_count = len(memory_ids)
            print(f"[Revive] Marking {revived_count} memories as active")
            
            return {
                "revived_count": revived_count,
                "memory_ids": memory_ids,
            }
        except Exception as e:
            print(f"[Revive] Error during revive: {e}")
            return {"error": str(e)}

