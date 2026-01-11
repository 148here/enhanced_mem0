from __future__ import annotations

import json
import socket
from typing import Any

import gradio as gr

from chat_manager import ChatManager
from config import AppConfig


print("=" * 50)
print("初始化配置...")
CFG = AppConfig()
print(f"配置加载完成: USER_ID={CFG.USER_ID}, MEMORY_DB_PATH={CFG.MEMORY_DB_PATH}")
print(f"WebUI 配置: host={CFG.WEBUI_HOST}, port={CFG.WEBUI_PORT}")

print("初始化 ChatManager...")
try:
    MANAGER = ChatManager(CFG)
    print("ChatManager 初始化完成")
except Exception as e:
    print(f"ChatManager 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    raise


def _status_markdown(status: dict[str, Any]) -> str:
    lines = [
        "### Current Status",
        f"- window_id: `{status.get('window_id')}`",
        f"- history_turns: **{status.get('history_turns')}** / max={status.get('max_history_turns')}",
        f"- top_k: **{status.get('top_k')}**",
        f"- last_search_count: **{status.get('last_search_count')}**",
        f"- last_add_success: **{status.get('last_add_success')}**",
        f"- chat_latency_s: **{status.get('chat_latency_s')}**",
        "",
        "### Memory Database Config",
        f"- provider: `{CFG.MEM0_VECTOR_STORE_PROVIDER}`",
        f"- path: `{CFG.MEMORY_DB_PATH}`",
        f"- collection: `{CFG.COLLECTION_NAME}`",
        f"- user_id: `{CFG.USER_ID}`",
        f"- embedding_dims: `{CFG.MEM0_EMBEDDING_DIMS}`",
        f"- model(chat): `{CFG.MODEL}`",
        f"- model(memory_llm): `{CFG.MEM0_MEMORY_LLM_MODEL}`",
        f"- embedding_model: `{CFG.EMBEDDING_MODEL}`",
    ]
    
    # 动态重要性和记忆衰减状态
    if status.get("dynamic_importance_enabled") is not None or status.get("fast_search_enabled") is not None:
        lines.extend([
            "",
            "### Dynamic Importance & Memory Decay Status",
            f"- dynamic_importance_enabled: **{status.get('dynamic_importance_enabled', False)}**",
            f"- fast_search_enabled: **{status.get('fast_search_enabled', False)}**",
            f"- revived_memories: **{status.get('revived_memories', 0)}**",
            f"- next_decay_in: **{status.get('next_decay_in', 'N/A')}** adds",
        ])
    
    # 如果启用裁判模型，添加相关状态
    if status.get("judge_enabled"):
        lines.extend([
            "",
            "### Judge Model + Dynamic TopK Status",
            f"- num_candidates: **{status.get('num_candidates')}**",
            f"- chosen_pick: **{status.get('chosen_pick')}**",
            f"- chosen_round: **{status.get('chosen_round')}**",
            f"- used_top_k: **{status.get('used_top_k')}**",
            f"- answer_time_s: **{status.get('answer_time_s')}**",
            f"- judge_time_s: **{status.get('judge_time_s')}**",
        ])
    
    return "\n".join(lines)


def _default_window() -> str:
    if MANAGER.list_windows():
        return MANAGER.list_windows()[0]
    return MANAGER.create_window()


def on_new_window(current_window_id: str | None):
    wid = MANAGER.create_window()
    choices = MANAGER.list_windows()
    # 新窗口：清空聊天显示
    return (
        gr.Dropdown(choices=choices, value=wid),
        [],
        _status_markdown(
            {
                "window_id": wid,
                "history_turns": 0,
                "max_history_turns": CFG.MAX_HISTORY_TURNS,
                "top_k": CFG.TOP_K,
                "last_search_count": 0,
                "last_add_success": None,
                "chat_latency_s": None,
                "memory_db_path": CFG.MEMORY_DB_PATH,
                "collection": CFG.COLLECTION_NAME,
                "judge_enabled": False,
                "dynamic_importance_enabled": CFG.ENABLE_DYNAMIC_IMPORTANCE,
                "fast_search_enabled": CFG.ENABLE_FAST_SEARCH,
                "revived_memories": 0,
                "next_decay_in": CFG.DECAY_CHECK_INTERVAL,
            }
        ),
        "",
        "[]",
        "",
        "[]",
        "{}",
        "[]",  # candidates
        "",    # judge_raw
    )


def on_switch_window(window_id: str):
    win = MANAGER.get_or_create_window(window_id)
    # Gradio Chatbot 需要 {"role": "user"/"assistant", "content": "..."} 格式
    messages = []
    for msg in win.history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    status = {
        "window_id": window_id,
        "history_turns": win.turns_count(),
        "max_history_turns": CFG.MAX_HISTORY_TURNS,
        "top_k": CFG.TOP_K,
        "last_search_count": 0,
        "last_add_success": None,
        "chat_latency_s": None,
        "memory_db_path": CFG.MEMORY_DB_PATH,
        "collection": CFG.COLLECTION_NAME,
        "judge_enabled": False,
        "dynamic_importance_enabled": CFG.ENABLE_DYNAMIC_IMPORTANCE,
        "fast_search_enabled": CFG.ENABLE_FAST_SEARCH,
        "revived_memories": 0,
        "next_decay_in": CFG.DECAY_CHECK_INTERVAL,
    }
    return messages, _status_markdown(status)


def on_send(window_id: str, user_input: str, chat_messages: list[dict[str, str]], enable_judge: bool, enable_importance: bool, enable_fast_search: bool):
    out = MANAGER.send_message(
        window_id, 
        user_input, 
        enable_judge_and_dynamic_topk=enable_judge,
        enable_dynamic_importance=enable_importance,
        enable_fast_search=enable_fast_search,
    )
    assistant = out["assistant_response"]

    # 更新聊天窗口显示 (Gradio Chatbot 格式)
    chat_messages = list(chat_messages or [])
    chat_messages.append({"role": "user", "content": user_input})
    chat_messages.append({"role": "assistant", "content": assistant})

    # 记忆提取展示
    mem = out["memory_extraction"]
    facts = mem.get("extracted_facts") or []
    raw = mem.get("raw_response") or ""
    mem0_result = mem.get("mem0_result") or {}

    # 裁判模型调试信息（如果启用）
    judge_debug = out.get("judge_debug", {})
    candidates_json = json.dumps(judge_debug.get("candidates", []), ensure_ascii=False, indent=2)
    judge_raw_text = judge_debug.get("judge_raw", "")

    return (
        chat_messages,
        _status_markdown(out["status"]),
        out["conversation_prompt"],
        json.dumps(out["retrieved_memories"], ensure_ascii=False, indent=2),
        raw,
        json.dumps(facts, ensure_ascii=False, indent=2),
        json.dumps(mem0_result, ensure_ascii=False, indent=2),
        candidates_json,  # 候选答案
        judge_raw_text,   # 裁判原始输出
        "",  # clear input
    )


with gr.Blocks(title="Mem0 在线对话（多窗口 + 共享记忆）") as demo:
    gr.Markdown("## Mem0 在线对话（多窗口 + 共享记忆）")

    with gr.Row():
        window_dd = gr.Dropdown(
            label="聊天窗口",
            choices=MANAGER.list_windows() or [_default_window()],
            value=_default_window(),
            interactive=True,
        )
        new_btn = gr.Button("新建窗口", variant="primary")

    chatbot = gr.Chatbot(label="对话", height=360)

    with gr.Row():
        user_tb = gr.Textbox(label="输入", placeholder="输入一段文本...", lines=2)
        send_btn = gr.Button("发送", variant="primary")

    # 裁判模型+动态topk 开关
    with gr.Row():
        enable_judge_cb = gr.Checkbox(
            label="Enable Judge Model + Dynamic TopK (increases latency but may improve answer quality)",
            value=CFG.ENABLE_JUDGE_AND_DYNAMIC_TOPK,
            interactive=True,
        )
    
    # 动态重要性和快速搜索开关
    with gr.Row():
        enable_importance_cb = gr.Checkbox(
            label="Enable Dynamic Importance (rerank search results by importance score)",
            value=CFG.ENABLE_DYNAMIC_IMPORTANCE,
            interactive=True,
        )
        enable_fast_search_cb = gr.Checkbox(
            label="Enable Fast Search (only search active/non-expired memories)",
            value=CFG.ENABLE_FAST_SEARCH,
            interactive=True,
        )

    status_md = gr.Markdown(_status_markdown({"window_id": _default_window(), "history_turns": 0, "max_history_turns": CFG.MAX_HISTORY_TURNS, "top_k": CFG.TOP_K, "last_search_count": 0, "last_add_success": None, "chat_latency_s": None, "memory_db_path": CFG.MEMORY_DB_PATH, "collection": CFG.COLLECTION_NAME}))

    with gr.Accordion("对话LLM：完整拼接内容（将发送给对话模型）", open=False):
        prompt_code = gr.Code(label="conversation_prompt", value="", language="markdown")
        retrieved_json = gr.Code(label="retrieved_memories (json)", value="[]", language="json")

    with gr.Accordion("add memory：记忆提取LLM原始返回", open=False):
        fact_raw_code = gr.Code(label="fact_extraction_raw", value="", language="json")
        facts_code = gr.Code(label="parsed_facts", value="[]", language="json")
        mem0_result_code = gr.Code(label="mem0.add() result", value="{}", language="json")

    # 裁判模型调试信息
    with gr.Accordion("Judge Model + Dynamic TopK: Debug Info (only available when enabled)", open=False):
        candidates_code = gr.Code(label="candidates (generated candidate answers)", value="[]", language="json")
        judge_raw_code = gr.Textbox(label="judge_raw (judge model raw output)", value="", lines=3)

    # events
    new_btn.click(
        fn=on_new_window,
        inputs=[window_dd],
        outputs=[window_dd, chatbot, status_md, prompt_code, retrieved_json, fact_raw_code, facts_code, mem0_result_code, candidates_code, judge_raw_code],
    )
    window_dd.change(fn=on_switch_window, inputs=[window_dd], outputs=[chatbot, status_md])

    send_btn.click(
        fn=on_send,
        inputs=[window_dd, user_tb, chatbot, enable_judge_cb, enable_importance_cb, enable_fast_search_cb],
        outputs=[chatbot, status_md, prompt_code, retrieved_json, fact_raw_code, facts_code, mem0_result_code, candidates_code, judge_raw_code, user_tb],
    )
    user_tb.submit(
        fn=on_send,
        inputs=[window_dd, user_tb, chatbot, enable_judge_cb, enable_importance_cb, enable_fast_search_cb],
        outputs=[chatbot, status_md, prompt_code, retrieved_json, fact_raw_code, facts_code, mem0_result_code, candidates_code, judge_raw_code, user_tb],
    )


def is_port_available(host: str, port: int) -> bool:
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


if __name__ == "__main__":
    print("=" * 50)
    print("开始启动 Mem0 在线对话 WebUI...")
    print(f"配置: host={CFG.WEBUI_HOST}, port={CFG.WEBUI_PORT}")
    print("=" * 50)

    try:
        print("初始化 Gradio demo...")
        demo.queue()
        print("Gradio demo 初始化完成")

        # 先检查端口是否可用
        print(f"检查端口 {CFG.WEBUI_PORT} 是否可用...")
        if not is_port_available(CFG.WEBUI_HOST, CFG.WEBUI_PORT):
            print(f"警告: 端口 {CFG.WEBUI_PORT} 被占用，尝试其他端口...")
            # 尝试 7861, 7862, ... 直到找到可用端口
            port = None
            for offset in range(1, 10):
                test_port = CFG.WEBUI_PORT + offset
                if is_port_available(CFG.WEBUI_HOST, test_port):
                    port = test_port
                    print(f"找到可用端口: {port}")
                    break
            if port is None:
                print("警告: 未找到可用端口，将让 Gradio 自动选择")
        else:
            port = CFG.WEBUI_PORT
            print(f"端口 {port} 可用")

        print(f"准备启动服务，使用端口: {port if port else '自动选择'}")
        if port:
            print(f"在端口 {port} 上启动服务...")
            demo.launch(server_name=CFG.WEBUI_HOST, server_port=port, share=False, inbrowser=False)
        else:
            # 让 Gradio 自动选择端口
            print("让 Gradio 自动选择端口...")
            demo.launch(server_name=CFG.WEBUI_HOST, share=False, inbrowser=False)
    except KeyboardInterrupt:
        print("\n用户中断，退出程序")
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        print("尝试使用 share=True 创建公共链接...")
        try:
            demo.launch(share=True, inbrowser=False)
        except Exception as e2:
            print(f"使用公共链接也失败: {e2}")
            import traceback
            traceback.print_exc()

