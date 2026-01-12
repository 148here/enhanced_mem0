import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH, JUDGE_PROMPT, JUDGE_PROMPT_GRAPH
from tqdm import tqdm

from mem0 import Memory

load_dotenv()

def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def _build_local_mem0(is_graph: bool, collection_name: Optional[str] = None) -> Memory:
    """
    本地 mem0（开源实现），替代托管版 MemoryClient。

    逻辑差异警告（很重要）：
    - 托管 API 用 `top_k`；本地 API 用 `limit`。
      在本文件中：我们把命令行参数 `top_k` 直接映射成 `limit=self.top_k`。
    - 托管 API 支持 `filter_memories` 参数；本地 `Memory.search()` 没有这个参数。
      所以 `--filter_memories` 在"本地 mem0"模式下不会生效（我们保留该 flag 以便兼容脚本）。
    - 托管 graph 返回 relations 里常见字段 `target`；本地 graph（Neo4j/Memgraph/Kuzu）常见是 `destination`。
      这里会做一次兼容映射：`target = relation.get("target", relation.get("destination"))`
    """
    vector_store_provider = os.getenv("MEM0_VECTOR_STORE_PROVIDER", "faiss")
    if collection_name is None:
        collection_name = os.getenv("MEM0_COLLECTION", "locomo_eval")
    embedding_dims = int(os.getenv("MEM0_EMBEDDING_DIMS", "1536"))

    config = {
        "version": "v1.1",
        "vector_store": {
            "provider": vector_store_provider,
            "config": {
                "collection_name": collection_name,
                "path": os.getenv("MEM0_VECTOR_STORE_PATH", os.path.join("results", "faiss")),
                "embedding_model_dims": embedding_dims,
            },
        },
        "embedder": {
            "provider": os.getenv("MEM0_EMBEDDER_PROVIDER", "openai"),
            "config": {
                "api_key": _require_env("OPENAI_API_KEY"),
                "model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            },
        },
        "llm": {
            "provider": os.getenv("MEM0_MEMORY_LLM_PROVIDER", "openai"),
            "config": {
                "api_key": _require_env("OPENAI_API_KEY"),
                "model": os.getenv("MEM0_MEMORY_LLM_MODEL", os.getenv("MODEL", "gpt-4o-mini")),
            },
        },
    }

    if is_graph:
        config["graph_store"] = {
            "provider": "neo4j",
            "config": {
                "url": _require_env("NEO4J_URL"),
                "username": _require_env("NEO4J_USERNAME"),
                "password": _require_env("NEO4J_PASSWORD"),
                "database": os.getenv("NEO4J_DATABASE") or None,
                "base_label": False,
            },
        }

    return Memory.from_config(config)


class MemorySearch:
    def __init__(
        self,
        output_path="results.json",
        top_k=10,
        filter_memories=False,
        is_graph=False,
        sample_rate=None,
        debug=False,
        num_candidates=1,
        max_expand_rounds=0,
        expand_step=4,
        candidate_temperature=0.2,
    ):
        # IMPORTANT: 这里改成"本地 mem0"，不走官方托管 API
        self.base_collection = os.getenv("MEM0_COLLECTION", "locomo_eval")
        self.mem0 = None  # 每个 conversation 单独加载一个 collection
        self.top_k = top_k
        self.openai_client = OpenAI()
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.sample_rate = sample_rate
        self.debug = debug
        self.num_candidates = num_candidates
        self.max_expand_rounds = max_expand_rounds
        self.expand_step = expand_step
        self.candidate_temperature = candidate_temperature

        # 排查 1: 打印库位置信息
        if self.debug:
            vector_store_path = os.getenv("MEM0_VECTOR_STORE_PATH", os.path.join("results", "faiss"))
            collection_name = os.getenv("MEM0_COLLECTION", "locomo_eval")
            print("=" * 80)
            print("[DEBUG - MemorySearch.__init__]")
            print(f"  os.getcwd(): {os.getcwd()}")
            print(f"  os.path.abspath(MEM0_VECTOR_STORE_PATH): {os.path.abspath(vector_store_path)}")
            print(f"  MEM0_COLLECTION: {collection_name}")
            print("=" * 80)

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
            self.JUDGE_PROMPT = JUDGE_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT
            self.JUDGE_PROMPT = JUDGE_PROMPT

    def search_memory(self, user_id, query, limit=None, max_retries=3, retry_delay=1):
        limit = self.top_k if limit is None else int(limit)
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                # NOTE:
                # - 旧：托管 API 用 top_k / filter_memories / enable_graph
                # - 新：本地 API 用 limit（=返回条数），graph 由 graph_store 配置决定
                if self.filter_memories:
                    # 本地 mem0 目前没有 filter_memories 参数；保留 flag 只是为了兼容 eval 脚本
                    # 如果你需要等价行为，需要你自己实现过滤逻辑（例如去重/阈值过滤/时间衰减）
                    pass
                # 排查 3: 确认 search 并没有"过滤掉 memory"
                # 当前实现：self.filter_memories 只是 pass，实际不做任何过滤
                memories = self.mem0.search(query, user_id=user_id, limit=limit)
                if self.debug:
                    semantic_results = memories.get("results", []) if isinstance(memories, dict) else memories
                    print(f"[DEBUG - MemorySearch.search_memory] user_id={user_id}, query={query[:50]}..., found {len(semantic_results)} memories")
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        # 本地 Memory.search(...) 返回 dict: {"results": [...], "relations": [...](可选)}
        semantic_results = memories.get("results", []) if isinstance(memories, dict) else memories
        semantic_memories = [
            {
                "memory": memory.get("memory"),
                "timestamp": (memory.get("metadata") or {}).get("timestamp"),
                "score": round(float(memory.get("score") or 0.0), 2),
            }
            for memory in (semantic_results or [])
        ]

        graph_memories = None
        if self.is_graph:
            relations = memories.get("relations") if isinstance(memories, dict) else None
            relations = relations or []
            graph_memories = [
                {
                    "source": relation.get("source"),
                    "relationship": relation.get("relationship"),
                    # 兼容：本地 graph 常用 destination；评测 prompt 里用 target
                    "target": relation.get("target", relation.get("destination")),
                }
                for relation in relations
            ]
        return semantic_memories, graph_memories, end_time - start_time

    def _normalize_one_line(self, text: str) -> str:
        resp = (text or "").strip()
        resp = resp.splitlines()[0].strip()
        norm = resp.strip().strip('"').strip("'").lower()
        norm = norm.rstrip(" \t\r\n.!?:;,。")
        if norm == "not specified":
            return ""
        return resp

    def _format_candidates_block(self, candidates: list[str]) -> str:
        # Candidates:
        # 1) ...
        # 2) ...
        lines = []
        for i, c in enumerate(candidates, 1):
            c = c.replace("\n", " ").strip()
            lines.append(f"{i}) {c}")
        return "\n".join(lines)

    def _generate_candidates(self, answer_prompt: str) -> tuple[list[str], float]:
        t1 = time.time()
        # 尽量一次请求拿到 n 个 candidates；如果 SDK 不支持 n，就 fallback 循环
        try:
            resp = self.openai_client.chat.completions.create(
                model=os.getenv("MODEL"),
                messages=[{"role": "system", "content": answer_prompt}],
                temperature=float(self.candidate_temperature),
                n=int(self.num_candidates),
            )
            raw = [(ch.message.content or "") for ch in resp.choices]
        except TypeError:
            raw = []
            for _ in range(int(self.num_candidates)):
                r = self.openai_client.chat.completions.create(
                    model=os.getenv("MODEL"),
                    messages=[{"role": "system", "content": answer_prompt}],
                    temperature=float(self.candidate_temperature),
                )
                raw.append(r.choices[0].message.content or "")
        t2 = time.time()
        cands = [self._normalize_one_line(x) for x in raw]
        # 防止全空：至少保留一个空串（后面 judge 会判 0）
        if not cands:
            cands = [""]
        return cands, (t2 - t1)

    def _judge_pick(self, judge_prompt: str, num_candidates: int) -> tuple[int, float, str]:
        t1 = time.time()
        r = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=[{"role": "system", "content": judge_prompt}],
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

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        first_round_first_candidate = None
        last_candidates = None

        total_answer_time = 0.0
        total_judge_time = 0.0

        chosen_pick = 1
        chosen_round = 0
        used_top_k = self.top_k
        judge_raw = ""

        # 初始化变量，确保回退时可用
        speaker_1_memories = []
        speaker_2_memories = []
        speaker_1_graph_memories = None
        speaker_2_graph_memories = None
        speaker_1_memory_time = 0.0
        speaker_2_memory_time = 0.0

        # 0..max_expand_rounds 轮
        for r in range(int(self.max_expand_rounds) + 1):
            cur_k = int(self.top_k) + r * int(self.expand_step)
            used_top_k = cur_k

            # 1) 检索（本轮用 cur_k）
            speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
                speaker_1_user_id, question, limit=cur_k
            )
            speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
                speaker_2_user_id, question, limit=cur_k
            )

            search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
            search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

            # 2) 生成 candidates（同一个 answer prompt，但 temperature>0 + n=num_candidates）
            template = Template(self.ANSWER_PROMPT)
            answer_prompt = template.render(
                speaker_1_user_id=speaker_1_user_id.split("_")[0],
                speaker_2_user_id=speaker_2_user_id.split("_")[0],
                speaker_1_memories=json.dumps(search_1_memory, indent=4),
                speaker_2_memories=json.dumps(search_2_memory, indent=4),
                speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
                speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
                question=question,
            )

            candidates, ans_time = self._generate_candidates(answer_prompt)
            total_answer_time += ans_time
            last_candidates = candidates

            if first_round_first_candidate is None:
                first_round_first_candidate = candidates[0] if candidates else ""

            # 特殊规则：当 num_candidates=1 时，跳过裁判步骤
            if self.num_candidates == 1:
                chosen_pick = 1
                chosen_round = r
                final_resp = candidates[0] if candidates else ""
                return (
                    final_resp,
                    speaker_1_memories,
                    speaker_2_memories,
                    speaker_1_memory_time,
                    speaker_2_memory_time,
                    speaker_1_graph_memories,
                    speaker_2_graph_memories,
                    total_answer_time,
                    total_judge_time,
                    candidates,
                    chosen_pick,
                    chosen_round,
                    used_top_k,
                    judge_raw,
                )

            # 3) 裁判选择序号（只看 memories + candidates）
            judge_tmpl = Template(self.JUDGE_PROMPT)
            judge_prompt = judge_tmpl.render(
                speaker_1_user_id=speaker_1_user_id.split("_")[0],
                speaker_2_user_id=speaker_2_user_id.split("_")[0],
                speaker_1_memories=json.dumps(search_1_memory, indent=4),
                speaker_2_memories=json.dumps(search_2_memory, indent=4),
                speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
                speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
                question=question,
                num_candidates=len(candidates),
                candidates=self._format_candidates_block(candidates),
            )

            pick, judge_time, judge_raw = self._judge_pick(judge_prompt, num_candidates=len(candidates))
            total_judge_time += judge_time

            if pick >= 1:
                chosen_pick = pick
                chosen_round = r
                final_resp = candidates[pick - 1]
                return (
                    final_resp,
                    speaker_1_memories,
                    speaker_2_memories,
                    speaker_1_memory_time,
                    speaker_2_memory_time,
                    speaker_1_graph_memories,
                    speaker_2_graph_memories,
                    total_answer_time,
                    total_judge_time,
                    candidates,
                    chosen_pick,
                    chosen_round,
                    used_top_k,
                    judge_raw,
                )

            # pick == 0 -> expand，继续下一轮

        # 所有轮次都 0：回退到"第一轮的第一个候选"
        final_resp = first_round_first_candidate or (last_candidates[0] if last_candidates else "")
        return (
            final_resp,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            total_answer_time,
            total_judge_time,
            last_candidates or [final_resp],
            1,  # fallback pick
            int(self.max_expand_rounds),
            used_top_k,
            judge_raw,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            answer_time,
            judge_time,
            candidates,
            chosen_pick,
            chosen_round,
            used_top_k,
            judge_raw,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": answer_time + judge_time,  # 保持向后兼容，但实际是 answer_time + judge_time
            "candidates": candidates,
            "chosen_pick": chosen_pick,
            "chosen_round": chosen_round,
            "used_top_k": used_top_k,
            "answer_time": answer_time,
            "judge_time": judge_time,
            "judge_raw": judge_raw,
        }

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path, max_conversations=None):
        with open(file_path, "r") as f:
            data = json.load(f)

        data_to_process = data[:max_conversations] if max_conversations else data
        
        # Calculate total questions for progress bar
        total_questions = sum(len(item["qa"]) for item in data_to_process)
        processed_questions = 0
        question_index = 0  # Global question index for sampling

        pbar = tqdm(total=total_questions, desc="Processing all questions")
        
        for idx, item in enumerate(data_to_process):
            qa = item["qa"]
            collection_name = f"{self.base_collection}_conv{idx}"
            self.mem0 = _build_local_mem0(is_graph=self.is_graph, collection_name=collection_name)

            # Debug: 打印 FAISS 内部状态（ntotal / index_to_id / docstore）
            if self.debug:
                vs = self.mem0.vector_store
                if hasattr(vs, 'index') and hasattr(vs, 'index_to_id') and hasattr(vs, 'docstore'):
                    print("FAISS ntotal=", vs.index.ntotal if vs.index is not None else 0,
                          " index_to_id=", len(vs.index_to_id),
                          " docstore=", len(vs.docstore))

            # 可选但强烈建议：没跑 add 的话直接报错（避免默默搜到空）
            if os.getenv("MEM0_VECTOR_STORE_PROVIDER", "faiss") == "faiss":
                vector_store_path = os.getenv("MEM0_VECTOR_STORE_PATH", os.path.join("results", "faiss"))
                faiss_fp = os.path.join(vector_store_path, f"{collection_name}.faiss")
                if not os.path.exists(faiss_fp):
                    raise FileNotFoundError(
                        f"Missing FAISS index for idx={idx}: {faiss_fp}. Did you run --method add first?"
                    )

            if self.debug:
                print(f"[DEBUG - MemorySearch.process_data_file] idx={idx}, collection={collection_name}")

            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"
            
            # 排查 2: 使用 get_all 直接数"写进去了多少条"
            if self.debug and idx < 2:  # 只检查前两个 conversation，避免输出过多
                try:
                    all_a = self.mem0.get_all(user_id=speaker_a_user_id, limit=10000)
                    all_b = self.mem0.get_all(user_id=speaker_b_user_id, limit=10000)
                    count_a = len(all_a.get("results", [])) if isinstance(all_a, dict) else len(all_a)
                    count_b = len(all_b.get("results", [])) if isinstance(all_b, dict) else len(all_b)
                    print("=" * 80)
                    print(f"[DEBUG - MemorySearch.process_data_file] Conversation {idx}")
                    print(f"  speaker_a_user_id: {speaker_a_user_id}")
                    print(f"  speaker_b_user_id: {speaker_b_user_id}")
                    print(f"  {speaker_a_user_id} memories count: {count_a}")
                    print(f"  {speaker_b_user_id} memories count: {count_b}")
                    print("=" * 80)
                except Exception as e:
                    print(f"[DEBUG - MemorySearch.process_data_file] Error getting all memories: {e}")

            for question_item in qa:
                # Apply sampling: if sample_rate is set, only process questions where index % sample_rate == 0
                if self.sample_rate is not None and question_index % self.sample_rate != 0:
                    question_index += 1
                    pbar.update(1)
                    continue
                
                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                self.results[idx].append(result)
                processed_questions += 1
                question_index += 1
                pbar.update(1)

                # Save results after each question is processed
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        pbar.close()

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results
