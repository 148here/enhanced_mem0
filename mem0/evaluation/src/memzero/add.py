import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import Memory
from prompts import CUSTOM_FACT_EXTRACTION_PROMPT_LOCOMO

load_dotenv()

# 多线程下 tqdm 输出需要全局锁，否则会出现进度条互相覆盖/乱跳
tqdm.set_lock(threading.RLock())


def _reset_faiss_collection(collection_name: str) -> None:
    """删除某个 collection 对应的本地 FAISS 文件（.faiss/.pkl），确保干净重跑。"""
    if os.getenv("MEM0_VECTOR_STORE_PROVIDER", "faiss") != "faiss":
        return
    vector_store_path = os.getenv("MEM0_VECTOR_STORE_PATH", os.path.join("results", "faiss"))
    os.makedirs(vector_store_path, exist_ok=True)
    for ext in (".faiss", ".pkl"):
        fp = os.path.join(vector_store_path, f"{collection_name}{ext}")
        if os.path.exists(fp):
            os.remove(fp)


def _require_env(key: str) -> str:
    """读取环境变量，不存在就抛出明确错误（避免静默使用托管 API 的配置）。"""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def _build_local_mem0(is_graph: bool, collection_name: Optional[str] = None) -> Memory:
    """
    使用开源/本地 `mem0.Memory`，而不是官方托管 `MemoryClient`。

    关键差异（一定要注意）：
    - 这里不会再用 MEM0_API_KEY / MEM0_PROJECT_ID / MEM0_ORGANIZATION_ID。
    - `Memory.add(...)` 本地 API 不支持 `version="v2"` / `enable_graph=...` 这种托管 API 参数；
      graph 是否启用由 `graph_store` 配置是否存在决定。
    """
    # 本地向量库：默认用 FAISS（纯本地文件），你可以在 .env 里覆盖 provider/path/collection。
    vector_store_provider = os.getenv("MEM0_VECTOR_STORE_PROVIDER", "faiss")
    if collection_name is None:
        collection_name = os.getenv("MEM0_COLLECTION", "locomo_eval")
    embedding_dims = int(os.getenv("MEM0_EMBEDDING_DIMS", "1536"))

    config = {
        "version": "v1.1",
        "custom_fact_extraction_prompt": CUSTOM_FACT_EXTRACTION_PROMPT_LOCOMO,
        "vector_store": {
            "provider": vector_store_provider,
            "config": {
                "collection_name": collection_name,
                # Windows 下最好显式指定路径，避免默认 /tmp 之类路径造成困扰
                "path": os.getenv("MEM0_VECTOR_STORE_PATH", os.path.join("results", "faiss")),
                "embedding_model_dims": embedding_dims,
            },
        },
        # 注意：这里的 embedder/llm 用于“抽取/更新记忆”，不等于评测回答问题用的 MODEL
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
        # 本地 graph（mem0-plus）需要你自己跑 Neo4j，并安装 mem0 的 graph extra 依赖。
        # 如果你没配置好，就直接报错，避免你误以为“开了 is_graph 但其实没启用图检索”。
        config["graph_store"] = {
            "provider": "neo4j",
            "config": {
                "url": _require_env("NEO4J_URL"),
                "username": _require_env("NEO4J_USERNAME"),
                "password": _require_env("NEO4J_PASSWORD"),
                "database": os.getenv("NEO4J_DATABASE") or None,
                # base_label=False：不强制所有节点都打 __Entity__ label（按你需要可改）
                "base_label": False,
            },
        }

    return Memory.from_config(config)


class MemoryADD:
    # 给“子进度条”分配固定的线程槽位（最多 max_workers 个），避免为每个 session 创建无限多条 tqdm
    _thread_slot_lock = threading.Lock()
    _thread_slots: dict[int, int] = {}
    _next_slot: int = 0

    def __init__(self, data_path=None, batch_size=2, is_graph=False, debug=False):
        # IMPORTANT: 这里改成"本地 mem0"，不走官方托管 API
        self.base_collection = os.getenv("MEM0_COLLECTION", "locomo_eval")
        self.mem0 = None  # 每个 conversation 单独建一个 collection
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        self.debug = debug
        
        # 排查 1: 打印库位置信息
        if self.debug:
            vector_store_path = os.getenv("MEM0_VECTOR_STORE_PATH", os.path.join("results", "faiss"))
            collection_name = os.getenv("MEM0_COLLECTION", "locomo_eval")
            print("=" * 80)
            print("[DEBUG - MemoryADD.__init__]")
            print(f"  os.getcwd(): {os.getcwd()}")
            print(f"  os.path.abspath(MEM0_VECTOR_STORE_PATH): {os.path.abspath(vector_store_path)}")
            print(f"  MEM0_COLLECTION: {collection_name}")
            print("=" * 80)
        
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        for attempt in range(retries):
            try:
                # NOTE:
                # - 旧版托管 API: MemoryClient.add(..., version="v2", enable_graph=...)
                # - 本地 API: Memory.add(...) 不支持这些参数；graph 由 graph_store 配置决定
                _ = self.mem0.add(message, user_id=user_id, metadata=metadata)
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, pbar: Optional[tqdm] = None):
        """写入一个 speaker 的所有 batch；如果传了 pbar，则每写入一个 batch 就 update(1)。"""
        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})
            if pbar is not None:
                pbar.update(1)

    @classmethod
    def _get_thread_slot(cls, max_workers: int) -> int:
        """将当前线程绑定到 [0, max_workers) 的槽位，用于 tqdm position。"""
        tid = threading.get_ident()
        with cls._thread_slot_lock:
            slot = cls._thread_slots.get(tid)
            if slot is not None:
                return slot
            slot = cls._next_slot % max_workers
            cls._thread_slots[tid] = slot
            cls._next_slot += 1
            return slot

    @staticmethod
    def _iter_session_keys(conversation: dict) -> list[str]:
        # session key: conversation 里除 speaker_a/speaker_b 以及 *date*/ *timestamp* 之外的键
        keys = []
        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue
            keys.append(key)
        return keys

    def prepare_conversation_collection(self, idx: int) -> str:
        """为某个 conversation 初始化/重置本地 collection，并构建 self.mem0。"""
        collection_name = f"{self.base_collection}_conv{idx}"
        _reset_faiss_collection(collection_name)
        self.mem0 = _build_local_mem0(is_graph=self.is_graph, collection_name=collection_name)
        if self.debug:
            print(f"[DEBUG - MemoryADD.prepare_conversation_collection] idx={idx}, collection={collection_name}")
        return collection_name

    def process_session(self, item: dict, idx: int, session_key: str) -> None:
        """以 session 为粒度处理，允许多个 session 并行；FAISS 层已通过锁保证落盘/索引结构不被写坏。"""
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        date_time_key = session_key + "_date_time"
        timestamp = conversation[date_time_key]
        chats = conversation[session_key]

        messages = []
        messages_reverse = []
        for chat in chats:
            if chat["speaker"] == speaker_a:
                messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
            elif chat["speaker"] == speaker_b:
                messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
            else:
                raise ValueError(f"Unknown speaker: {chat['speaker']}")

        # 子进度条：并发情况下不可能只显示“一条”，合理做法是最多显示 max_workers 组（A/B 两条）。
        # 我们把每个线程绑定到一个固定槽位，复用该槽位显示当前正在处理的 session。
        max_workers = int(os.getenv("MEM0_SESSION_WORKERS", "10"))
        slot = self._get_thread_slot(max_workers=max_workers)
        pos_a = 1 + slot * 2
        pos_b = 2 + slot * 2

        total_a = (len(messages) + self.batch_size - 1) // self.batch_size
        total_b = (len(messages_reverse) + self.batch_size - 1) // self.batch_size

        desc_prefix = f"conv{idx}:{session_key}"
        with tqdm(total=total_a, position=pos_a, leave=False, desc=f"{desc_prefix} | Speaker A") as pbar_a:
            self.add_memories_for_speaker(speaker_a_user_id, messages, timestamp, pbar=pbar_a)
        with tqdm(total=total_b, position=pos_b, leave=False, desc=f"{desc_prefix} | Speaker B") as pbar_b:
            self.add_memories_for_speaker(speaker_b_user_id, messages_reverse, timestamp, pbar=pbar_b)

    def process_all_conversations(self, max_workers=10, max_conversations=None):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        
        data_to_process = self.data[:max_conversations] if max_conversations else self.data
        total_conversations = len(data_to_process)

        # 先为每个 conversation 准备独立 collection + Memory 实例（collection 仍是 conversation 级别隔离）
        workers: list[MemoryADD] = []
        session_tasks: list[tuple[MemoryADD, dict, int, str]] = []

        for idx, item in enumerate(data_to_process):
            worker = MemoryADD(
                data_path=None,
                batch_size=self.batch_size,
                is_graph=self.is_graph,
                debug=self.debug,
            )
            worker.prepare_conversation_collection(idx)
            workers.append(worker)

            conversation = item["conversation"]
            for session_key in self._iter_session_keys(conversation):
                session_tasks.append((worker, item, idx, session_key))

        total_sessions = len(session_tasks)

        # 让 session 任务的并发度也可通过环境变量统一控制（与子进度条槽位数一致）
        # - 不设置则保持原行为：使用函数参数 max_workers
        env_workers = os.getenv("MEM0_SESSION_WORKERS")
        if env_workers:
            try:
                max_workers = int(env_workers)
            except ValueError:
                pass

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(worker.process_session, item, idx, session_key)
                for (worker, item, idx, session_key) in session_tasks
            ]

            for future in tqdm(
                as_completed(futures),
                total=total_sessions,
                position=0,
                desc="Processing all sessions",
            ):
                future.result()

        # Debug：每个 conversation 结束后数一下写入条数（避免并发下“看似写了但实际没落盘”）
        if self.debug:
            for idx, item in enumerate(data_to_process):
                conversation = item["conversation"]
                speaker_a = conversation["speaker_a"]
                speaker_b = conversation["speaker_b"]
                speaker_a_user_id = f"{speaker_a}_{idx}"
                speaker_b_user_id = f"{speaker_b}_{idx}"
                try:
                    all_a = workers[idx].mem0.get_all(user_id=speaker_a_user_id, limit=100000)
                    all_b = workers[idx].mem0.get_all(user_id=speaker_b_user_id, limit=100000)
                    count_a = len(all_a.get("results", [])) if isinstance(all_a, dict) else len(all_a)
                    count_b = len(all_b.get("results", [])) if isinstance(all_b, dict) else len(all_b)
                    print("=" * 80)
                    print(f"[DEBUG - MemoryADD.process_all_conversations] Conversation {idx}")
                    print(f"  speaker_a_user_id: {speaker_a_user_id}")
                    print(f"  speaker_b_user_id: {speaker_b_user_id}")
                    print(f"  {speaker_a_user_id} memories count: {count_a}")
                    print(f"  {speaker_b_user_id} memories count: {count_b}")
                    print("=" * 80)
                except Exception as e:
                    print(f"[DEBUG - MemoryADD.process_all_conversations] Error getting all memories: {e}")

        print(f"Messages added successfully. conversations={total_conversations}, sessions={total_sessions}")
