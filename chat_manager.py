from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from jinja2 import Template
from openai import OpenAI

from config import AppConfig, require_env
from memory_system import MemorySystem
from prompt import ANSWER_PROMPT, JUDGE_PROMPT, MEMORY_ANSWER_PROMPT


@dataclass
class ChatWindow:
    window_id: str
    # 按顺序保存 role=user/assistant 的消息；我们保证每一轮都按 user -> assistant 追加
    history: list[dict[str, str]]

    def turns_count(self) -> int:
        return len(self.history) // 2

    def recent_messages_for_llm(self, *, max_turns: int) -> list[dict[str, str]]:
        # 1轮 = user + assistant；我们严格按对写入，所以直接截最后 2*max_turns 条即可
        if max_turns <= 0:
            return []
        return self.history[-2 * max_turns :]


class ChatManager:
    """
    多窗口在线对话管理：
    - 不同窗口：history 不共享
    - 同一用户：memory 共享（同一个 user_id / collection / faiss）
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        require_env("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=self.cfg.OPENAI_API_KEY)
        self.memory = MemorySystem(cfg)
        self.windows: dict[str, ChatWindow] = {}

        # 使用 mem0 官方的 MEMORY_ANSWER_PROMPT
        self.base_system_prompt = MEMORY_ANSWER_PROMPT

    def list_windows(self) -> list[str]:
        return list(self.windows.keys())

    def get_or_create_window(self, window_id: str) -> ChatWindow:
        if window_id not in self.windows:
            self.windows[window_id] = ChatWindow(window_id=window_id, history=[])
        return self.windows[window_id]

    def create_window(self) -> str:
        wid = f"window_{int(time.time() * 1000)}"
        self.get_or_create_window(wid)
        return wid

    def _format_memories_block(self, memories: list[dict[str, Any]]) -> str:
        if not memories:
            return "(none)"
        lines = []
        for i, m in enumerate(memories, 1):
            mem = (m.get("memory") or "").strip()
            score = m.get("score")
            ts = (m.get("metadata") or {}).get("timestamp")
            parts = [f"{i}. {mem}"]
            if score is not None:
                parts.append(f"(score={float(score):.4f})")
            if ts is not None:
                parts.append(f"(ts={ts})")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def _build_system_prompt(self, retrieved_memories: list[dict[str, Any]]) -> str:
        mem_block = self._format_memories_block(retrieved_memories)
        return (
            f"{self.base_system_prompt}\n\n"
            f"Here are the memories I have stored:\n{mem_block}\n\n"
            f"Please use these memories to provide accurate and concise answers to the questions."
        )

    # =========================
    # 裁判模型+动态topk 相关方法
    # =========================

    def _normalize_one_line(self, text: str) -> str:
        """规范化候选答案：只保留第一行，去除引号和多余空格"""
        resp = (text or "").strip()
        resp = resp.splitlines()[0].strip()
        norm = resp.strip().strip('"').strip("'").lower()
        norm = norm.rstrip(" \t\r\n.!?:;,。")
        if norm == "not specified":
            return ""
        return resp

    def _format_candidates_block(self, candidates: list[str]) -> str:
        """格式化候选答案列表为字符串，供裁判模型评估"""
        lines = []
        for i, c in enumerate(candidates, 1):
            c = c.replace("\n", " ").strip()
            lines.append(f"{i}) {c}")
        return "\n".join(lines)

    def _generate_candidates(self, answer_prompt: str) -> tuple[list[str], float]:
        """
        生成多个候选答案
        返回: (候选答案列表, 耗时)
        """
        t1 = time.time()
        try:
            # 尽量一次请求拿到 n 个 candidates；如果 SDK 不支持 n，就 fallback 循环
            resp = self.openai_client.chat.completions.create(
                model=self.cfg.MODEL,
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=float(self.cfg.CANDIDATE_TEMPERATURE),
                n=int(self.cfg.NUM_CANDIDATES),
            )
            raw = [(ch.message.content or "") for ch in resp.choices]
        except TypeError:
            # fallback: 循环调用
            raw = []
            for _ in range(int(self.cfg.NUM_CANDIDATES)):
                r = self.openai_client.chat.completions.create(
                    model=self.cfg.MODEL,
                    messages=[{"role": "user", "content": answer_prompt}],
                    temperature=float(self.cfg.CANDIDATE_TEMPERATURE),
                )
                raw.append(r.choices[0].message.content or "")
        t2 = time.time()
        cands = [self._normalize_one_line(x) for x in raw]
        # 防止全空：至少保留一个空串（后面 judge 会判 0）
        if not cands:
            cands = [""]
        return cands, (t2 - t1)

    def _judge_pick(self, judge_prompt: str, num_candidates: int) -> tuple[int, float, str]:
        """
        裁判模型选择最佳候选
        返回: (选择的序号 1..num_candidates 或 0, 耗时, 原始输出)
        """
        t1 = time.time()
        r = self.openai_client.chat.completions.create(
            model=self.cfg.MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
        )
        t2 = time.time()
        out = (r.choices[0].message.content or "").strip()
        # 鲁棒解析：抓第一个整数
        m = re.search(r"-?\d+", out)
        pick = int(m.group(0)) if m else 0
        if pick < 0 or pick > num_candidates:
            pick = 0
        return pick, (t2 - t1), out

    def _build_answer_prompt(
        self,
        retrieved_memories: list[dict[str, Any]],
        user_input: str,
        recent_history: list[dict[str, str]],
    ) -> str:
        """构建生成候选答案的 prompt（使用 Jinja2 Template）"""
        # 格式化记忆
        mem_lines = []
        for m in retrieved_memories:
            mem_text = (m.get("memory") or "").strip()
            score = m.get("score")
            ts = (m.get("metadata") or {}).get("timestamp")
            parts = [mem_text]
            if score is not None:
                parts.append(f"(score={float(score):.4f})")
            if ts is not None:
                parts.append(f"(ts={ts})")
            mem_lines.append(" ".join(parts))
        memories_str = "\n".join(mem_lines) if mem_lines else "(none)"

        # 格式化历史
        history_str = json.dumps(recent_history, ensure_ascii=False, indent=2) if recent_history else "(none)"

        # 渲染 template
        template = Template(ANSWER_PROMPT)
        return template.render(
            memories=memories_str,
            history=history_str,
            question=user_input,
        )

    def _build_judge_prompt(
        self,
        retrieved_memories: list[dict[str, Any]],
        user_input: str,
        recent_history: list[dict[str, str]],
        candidates: list[str],
    ) -> str:
        """构建裁判评估的 prompt（使用 Jinja2 Template）"""
        # 格式化记忆（同上）
        mem_lines = []
        for m in retrieved_memories:
            mem_text = (m.get("memory") or "").strip()
            score = m.get("score")
            ts = (m.get("metadata") or {}).get("timestamp")
            parts = [mem_text]
            if score is not None:
                parts.append(f"(score={float(score):.4f})")
            if ts is not None:
                parts.append(f"(ts={ts})")
            mem_lines.append(" ".join(parts))
        memories_str = "\n".join(mem_lines) if mem_lines else "(none)"

        # 格式化历史
        history_str = json.dumps(recent_history, ensure_ascii=False, indent=2) if recent_history else "(none)"

        # 格式化候选
        candidates_str = self._format_candidates_block(candidates)

        # 渲染 template
        template = Template(JUDGE_PROMPT)
        return template.render(
            memories=memories_str,
            history=history_str,
            question=user_input,
            num_candidates=len(candidates),
            candidates=candidates_str,
        )

    def _build_prompt_debug_text(
        self,
        *,
        system_prompt: str,
        recent_history: list[dict[str, str]],
        user_input: str,
        retrieved_memories: list[dict[str, Any]],
    ) -> str:
        return (
            "=== System ===\n"
            f"{system_prompt}\n\n"
            "=== Retrieved Memories (raw json) ===\n"
            f"{json.dumps(retrieved_memories, ensure_ascii=False, indent=2)}\n\n"
            "=== Recent History (messages) ===\n"
            f"{json.dumps(recent_history, ensure_ascii=False, indent=2)}\n\n"
            "=== User Input ===\n"
            f"{user_input}\n"
        )

    def send_message(
        self, 
        window_id: str, 
        user_input: str, 
        enable_judge_and_dynamic_topk: bool = False,
        enable_dynamic_importance: bool = False,
        enable_fast_search: bool = False,
    ) -> dict[str, Any]:
        """
        发送消息并获取回复
        
        Args:
            window_id: 窗口ID
            user_input: 用户输入
            enable_judge_and_dynamic_topk: 是否启用裁判模型+动态topk机制
            enable_dynamic_importance: 是否启用动态重要性排序
            enable_fast_search: 是否只搜索活跃记忆
        
        Returns:
            包含回复和调试信息的字典
        """
        win = self.get_or_create_window(window_id)
        user_input = (user_input or "").strip()
        if not user_input:
            raise ValueError("Empty user input")

        # 如果启用裁判模型+动态topk，使用新流程（强制禁用快速搜索）
        if enable_judge_and_dynamic_topk:
            return self._send_message_with_judge(window_id, user_input, win, enable_dynamic_importance, False)
        
        # 否则使用原有流程（直接生成）
        return self._send_message_direct(window_id, user_input, win, enable_dynamic_importance, enable_fast_search)

    def _send_message_direct(
        self, 
        window_id: str, 
        user_input: str, 
        win: ChatWindow,
        enable_dynamic_importance: bool,
        enable_fast_search: bool,
    ) -> dict[str, Any]:
        """原有的直接生成流程（无裁判模型）"""
        # 1) memory search（共享）- 使用新参数
        retrieved_memories = self.memory.search(
            user_input, 
            top_k=self.cfg.TOP_K,
            use_fast_search=enable_fast_search,
            enable_dynamic_importance=enable_dynamic_importance,
        )

        # 2) history（窗口隔离 + 最近N轮可见）
        recent_history = win.recent_messages_for_llm(max_turns=self.cfg.MAX_HISTORY_TURNS)

        # 3) build messages for OpenAI
        system_prompt = self._build_system_prompt(retrieved_memories)
        llm_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        llm_messages.extend(recent_history)
        llm_messages.append({"role": "user", "content": user_input})

        prompt_debug_text = self._build_prompt_debug_text(
            system_prompt=system_prompt,
            recent_history=recent_history,
            user_input=user_input,
            retrieved_memories=retrieved_memories,
        )

        # 4) call chat LLM
        chat_t1 = time.time()
        resp = self.openai_client.chat.completions.create(
            model=self.cfg.MODEL,
            messages=llm_messages,
            temperature=0.2,
        )
        chat_t2 = time.time()
        assistant_text = (resp.choices[0].message.content or "").strip()

        # 5) update history (window)
        win.history.append({"role": "user", "content": user_input})
        win.history.append({"role": "assistant", "content": assistant_text})

        # 6) revive used memories
        used_memory_ids = [m.get("id", "") for m in retrieved_memories if m.get("id")]
        revive_result = self.memory.revive_memories(used_memory_ids) if used_memory_ids else {}

        # 7) add memory (shared) + capture fact extraction raw
        mem_dbg = self.memory.add_turn(user_input, assistant_text)

        # 获取衰减统计
        decay_info = self.memory.debug_state()

        status = {
            "window_id": window_id,
            "history_turns": win.turns_count(),
            "max_history_turns": self.cfg.MAX_HISTORY_TURNS,
            "top_k": self.cfg.TOP_K,
            "memory_db_path": self.cfg.MEMORY_DB_PATH,
            "collection": self.cfg.COLLECTION_NAME,
            "last_search_count": len(retrieved_memories),
            "last_add_success": bool(mem_dbg.mem0_result) and not mem_dbg.error,
            "chat_latency_s": round(chat_t2 - chat_t1, 3),
            # 动态重要性相关
            "dynamic_importance_enabled": enable_dynamic_importance,
            "fast_search_enabled": enable_fast_search,
            "revived_memories": revive_result.get("revived_count", 0),
            "next_decay_in": decay_info.get("next_decay_in", 0),
        }

        return {
            "assistant_response": assistant_text,
            "conversation_prompt": prompt_debug_text,
            "memory_extraction": {
                "extracted_facts": mem_dbg.extracted_facts or [],
                "raw_response": mem_dbg.fact_extraction_raw or "",
                "error": mem_dbg.error,
                "mem0_result": mem_dbg.mem0_result,
            },
            "retrieved_memories": retrieved_memories,
            "recent_history": recent_history,
            "status": status,
        }

    def _send_message_with_judge(
        self, 
        window_id: str, 
        user_input: str, 
        win: ChatWindow,
        enable_dynamic_importance: bool,
        enable_fast_search: bool,
    ) -> dict[str, Any]:
        """裁判模型+动态topk流程（强制全局搜索，不使用快速搜索）"""
        first_round_first_candidate = None
        last_candidates = None

        total_answer_time = 0.0
        total_judge_time = 0.0

        chosen_pick = 1
        chosen_round = 0
        used_top_k = self.cfg.TOP_K
        judge_raw = ""

        # 初始化变量（确保回退时可用）
        retrieved_memories = []
        recent_history = win.recent_messages_for_llm(max_turns=self.cfg.MAX_HISTORY_TURNS)

        # 循环 0..max_expand_rounds 轮
        for r in range(int(self.cfg.MAX_EXPAND_ROUNDS) + 1):
            cur_k = int(self.cfg.TOP_K) + r * int(self.cfg.EXPAND_STEP)
            used_top_k = cur_k

            # 1) 检索记忆（本轮用 cur_k，强制全局搜索）
            retrieved_memories = self.memory.search(
                user_input, 
                top_k=cur_k,
                use_fast_search=False,  # 裁判模型强制全局搜索
                enable_dynamic_importance=enable_dynamic_importance,
            )

            # 2) 生成候选答案
            answer_prompt = self._build_answer_prompt(retrieved_memories, user_input, recent_history)
            candidates, ans_time = self._generate_candidates(answer_prompt)
            total_answer_time += ans_time
            last_candidates = candidates

            if first_round_first_candidate is None:
                first_round_first_candidate = candidates[0] if candidates else ""

            # 3) 特殊规则：num_candidates=1 时跳过裁判步骤
            if self.cfg.NUM_CANDIDATES == 1:
                chosen_pick = 1
                chosen_round = r
                final_resp = candidates[0] if candidates else ""
                break

            # 4) 裁判选择
            judge_prompt = self._build_judge_prompt(retrieved_memories, user_input, recent_history, candidates)
            pick, judge_time, judge_raw = self._judge_pick(judge_prompt, num_candidates=len(candidates))
            total_judge_time += judge_time

            # 5) 如果裁判选择有效候选（pick >= 1），返回结果
            if pick >= 1:
                chosen_pick = pick
                chosen_round = r
                final_resp = candidates[pick - 1]
                break

            # 6) pick == 0 -> expand，继续下一轮
        else:
            # 所有轮次都 0：回退到"第一轮的第一个候选"
            final_resp = first_round_first_candidate or (last_candidates[0] if last_candidates else "")
            chosen_pick = 1  # fallback pick
            chosen_round = int(self.cfg.MAX_EXPAND_ROUNDS)

        # 7) 更新历史
        assistant_text = final_resp
        win.history.append({"role": "user", "content": user_input})
        win.history.append({"role": "assistant", "content": assistant_text})

        # 8) revive used memories
        used_memory_ids = [m.get("id", "") for m in retrieved_memories if m.get("id")]
        revive_result = self.memory.revive_memories(used_memory_ids) if used_memory_ids else {}

        # 9) 添加记忆
        mem_dbg = self.memory.add_turn(user_input, assistant_text)

        # 10) 获取衰减统计
        decay_info = self.memory.debug_state()

        # 11) 构建调试信息
        prompt_debug_text = self._build_prompt_debug_text(
            system_prompt=f"[Judge Mode] Round {chosen_round}, Pick {chosen_pick}/{self.cfg.NUM_CANDIDATES}",
            recent_history=recent_history,
            user_input=user_input,
            retrieved_memories=retrieved_memories,
        )

        status = {
            "window_id": window_id,
            "history_turns": win.turns_count(),
            "max_history_turns": self.cfg.MAX_HISTORY_TURNS,
            "top_k": self.cfg.TOP_K,
            "memory_db_path": self.cfg.MEMORY_DB_PATH,
            "collection": self.cfg.COLLECTION_NAME,
            "last_search_count": len(retrieved_memories),
            "last_add_success": bool(mem_dbg.mem0_result) and not mem_dbg.error,
            "chat_latency_s": round(total_answer_time + total_judge_time, 3),
            # 裁判模型相关状态
            "judge_enabled": True,
            "num_candidates": self.cfg.NUM_CANDIDATES,
            "chosen_pick": chosen_pick,
            "chosen_round": chosen_round,
            "used_top_k": used_top_k,
            "answer_time_s": round(total_answer_time, 3),
            "judge_time_s": round(total_judge_time, 3),
            # 动态重要性相关
            "dynamic_importance_enabled": enable_dynamic_importance,
            "fast_search_enabled": False,  # 裁判模式强制全局搜索
            "revived_memories": revive_result.get("revived_count", 0),
            "next_decay_in": decay_info.get("next_decay_in", 0),
        }

        return {
            "assistant_response": assistant_text,
            "conversation_prompt": prompt_debug_text,
            "memory_extraction": {
                "extracted_facts": mem_dbg.extracted_facts or [],
                "raw_response": mem_dbg.fact_extraction_raw or "",
                "error": mem_dbg.error,
                "mem0_result": mem_dbg.mem0_result,
            },
            "retrieved_memories": retrieved_memories,
            "recent_history": recent_history,
            "status": status,
            # 裁判模型相关调试信息
            "judge_debug": {
                "candidates": last_candidates or [],
                "judge_raw": judge_raw,
                "chosen_pick": chosen_pick,
                "chosen_round": chosen_round,
                "used_top_k": used_top_k,
            },
        }

