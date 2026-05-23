"""
LM Studio Chat - FastAPI アプリケーション（Corpus2Skill 版）
"""

import asyncio
import httpx
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from mem0 import AsyncMemory
    _MEM0_AVAILABLE = True
except ImportError:
    _MEM0_AVAILABLE = False

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config
from rag import Corpus2SkillManager


# ─── アプリケーション設定 ───────────────────────────────────────────────

app = FastAPI(title="LM Studio Chat (Corpus2Skill)")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

HISTORY_FILE = BASE_DIR / "chat_history.json"

# ─── Corpus2Skill 初期化 ─────────────────────────────────────────────────

RAG_DOCS_DIR    = BASE_DIR / "rag_docs"
RAG_CONFIG_FILE = BASE_DIR / "rag_config.json"
RAG_DOCS_DIR.mkdir(exist_ok=True)


def load_rag_config() -> dict:
    """rag_config.json を読み込む。存在しない場合はデフォルト値を返す"""
    defaults = {
        "llm_model": "",
        "max_top_skills": 6,
        "branching_factor": 4,
        "chunk_max_chars": 800,
    }
    if RAG_CONFIG_FILE.exists():
        try:
            cfg = json.loads(RAG_CONFIG_FILE.read_text(encoding="utf-8"))
            return {
                "llm_model":       cfg.get("llm_model", ""),
                "max_top_skills":  int(cfg.get("max_top_skills", 6)),
                "branching_factor": int(cfg.get("branching_factor", 4)),
                "chunk_max_chars": int(cfg.get("chunk_max_chars", 800)),
            }
        except Exception:
            pass
    return defaults


rag_manager: Corpus2SkillManager | None = None
mem0_memory: "AsyncMemory | None" = None


def _get_embedding_dims(model_name: str) -> int:
    """sentence-transformers モデルの埋め込み次元数を取得する"""
    _known = {
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "paraphrase-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-mpnet-base-v2": 768,
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
    }
    short_name = model_name.split("/")[-1]
    return _known.get(short_name, _known.get(model_name, 384))


def _create_mem0_config() -> dict:
    host = os.getenv("LM_STUDIO_HOST", "127.0.0.1")
    port = os.getenv("LM_STUDIO_PORT", "1234")
    base_url = f"http://{host}:{port}"
    api_key = os.getenv("LM_STUDIO_API_KEY", "") or "lm-studio"
    cfg = load_rag_config()
    llm_model = cfg.get("llm_model") or os.getenv("LM_STUDIO_LLM_MODEL", "") or "local"
    embed_model = os.getenv(
        "C2S_EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embedding_dims = _get_embedding_dims(embed_model)
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "api_key": api_key,
                "openai_base_url": f"{base_url}/v1",
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": embed_model,
                "embedding_dims": embedding_dims,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "chat_memories",
                "path": str(BASE_DIR / "mem0_db"),
                "embedding_model_dims": embedding_dims,
            },
        },
    }


def _get_user_id(session: str) -> str:
    parts = session.split("_", 2)
    if len(parts) >= 2 and parts[0] == "user":
        return parts[1]
    return "default"


def _parse_mem0_results(result) -> list[dict]:
    if isinstance(result, dict):
        return result.get("results", [])
    if isinstance(result, list):
        return result
    return []


@app.on_event("startup")
async def startup_event():
    global rag_manager, mem0_memory
    try:
        cfg = load_rag_config()
        base_url = (
            f"http://{os.getenv('LM_STUDIO_HOST', '127.0.0.1')}"
            f":{os.getenv('LM_STUDIO_PORT', '1234')}/v1"
        )
        # LLM モデルの優先順位: rag_config.json（UI保存値）> 環境変数 > 自動検出
        llm_model = cfg.get("llm_model", "") or os.getenv("LM_STUDIO_LLM_MODEL", "")

        rag_manager = Corpus2SkillManager(
            working_dir=str(BASE_DIR / "c2s_db"),
            lm_studio_base_url=base_url,
            llm_model=llm_model,
            embed_model=os.getenv(
                "C2S_EMBED_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
            max_top_skills=cfg.get("max_top_skills", 6),
            branching_factor=cfg.get("branching_factor", 4),
            chunk_max_chars=cfg.get("chunk_max_chars", 800),
        )
        await rag_manager.initialize()
        print(f"[Corpus2Skill] 初期化完了 - {rag_manager.get_status()}")
    except Exception as e:
        print(f"[Corpus2Skill] 初期化失敗（RAG機能は無効）: {e}")
        rag_manager = None

    if _MEM0_AVAILABLE:
        try:
            mem0_memory = AsyncMemory.from_config(_create_mem0_config())
            print("[mem0] 初期化完了")
        except Exception as e:
            print(f"[mem0] 初期化失敗（メモリ機能は無効）: {e}")
            mem0_memory = None


# ─── モデル定義 ─────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 8196


class LoginRequest(BaseModel):
    username: str
    password: str


class SettingsUpdate(BaseModel):
    lm_studio_host: str
    lm_studio_port: int
    app_username: str
    app_password: str
    lm_studio_api_key: str


# ─── チャット履歴管理 ───────────────────────────────────────────────────


def load_history() -> dict:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_history(history: dict) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_session_history(session_id: str) -> list[dict]:
    history = load_history()
    return history.get(session_id, [])


def append_to_history(session_id: str, role: str, content: str) -> None:
    history = load_history()
    if session_id not in history:
        history[session_id] = []

    history[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

    if len(history[session_id]) > 100:
        history[session_id] = history[session_id][-100:]

    save_history(history)


def clear_session_history(session_id: str) -> None:
    history = load_history()
    if session_id in history:
        del history[session_id]
    save_history(history)


# ─── LM Studio API 通信 ─────────────────────────────────────────────────


async def fetch_models() -> list[dict]:
    try:
        api_key = config.Config.get_api_key()
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                config.Config.get_models_endpoint(),
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
    except Exception as e:
        print(f"モデル取得エラー: {e}")
        return []


async def send_to_lm_studio(
    messages: list[dict],
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8196
) -> str:
    api_key = config.Config.get_api_key()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        payload["model"] = model

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                config.Config.get_api_endpoint(),
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        return "⚠️ タイムアウトしました。LM Studio サーバーが実行中か確認してください。"
    except httpx.HTTPStatusError as e:
        return f"⚠️ HTTPエラー: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"⚠️ エラーが発生しました: {str(e)}"


# ─── 認証 ───────────────────────────────────────────────────────────────


def check_auth(request: Request) -> Optional[str]:
    session = request.cookies.get("session_token")
    if not session:
        return None
    return session


# ─── ルート ─────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    session = request.cookies.get("session_token")

    if session:
        return templates.TemplateResponse(
            name="chat.html",
            context={
                "authenticated": True,
                "session_id": session,
                "history": get_session_history(session),
                "models": await fetch_models(),
                "lm_studio_url": config.Config.get_lm_studio_url(),
            },
            request=request
        )

    return templates.TemplateResponse(
        name="login.html",
        context={
            "authenticated": False,
            "error": "ユーザー名またはパスワードが異なります。",
        },
        request=request
    )


@app.post("/login")
async def login(request: Request):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")

    if config.Config.is_authenticated(username, password):
        response = HTMLResponse(
            """<script>window.location.href='/';</script>"""
        )
        response.set_cookie(
            key="session_token",
            value=f"user_{username}_{datetime.now().timestamp()}",
            httponly=True,
            max_age=86400,
        )
        return response

    return templates.TemplateResponse("login.html", {
        "request": request,
        "authenticated": False,
        "error": "ユーザー名またはパスワードが異なります。",
    })


@app.get("/logout")
async def logout(request: Request):
    response = HTMLResponse("""<script>window.location.href='/';</script>""")
    response.delete_cookie(key="session_token")
    return response


@app.get("/api/models")
async def api_models(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    models = await fetch_models()
    return JSONResponse({"models": models})


@app.post("/api/chat")
async def api_chat(request: Request):
    """チャットリクエスト API（ストリーミング SSE + Corpus2Skill RAG + ツール対応版）"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    messages    = body.get("messages", [])
    model       = body.get("model", "")
    temperature = body.get("temperature", 0.7)
    max_tokens  = body.get("max_tokens", 8192)
    tools       = body.get("tools", [])
    use_rag     = body.get("use_rag", False)
    use_memory  = body.get("use_memory", False)

    if not messages:
        return JSONResponse({"error": "メッセージが空です"}, status_code=400)

    # ユーザーの発言を先に取得（履歴保存用）
    user_content = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )

    async def generate():
        chat_messages = list(messages)
        rag_sources: list[dict] = []
        memory_context: list[dict] = []

        # ── メモリ検索 ──────────────────────────────────────────────────
        if use_memory and mem0_memory and user_content:
            yield f"data: {json.dumps({'type': 'status', 'message': '🧠 メモリ検索中...'})}\n\n"
            try:
                user_id = _get_user_id(session)
                results = await mem0_memory.search(
                    user_content, top_k=5, filters={"user_id": user_id}
                )
                memory_context = _parse_mem0_results(results)
                if memory_context:
                    memory_text = "\n".join(
                        f"- {m.get('memory', m.get('text', ''))}"
                        for m in memory_context
                    )
                    chat_messages = [{
                        "role": "system",
                        "content": f"【会話の記憶】\n{memory_text}",
                    }] + chat_messages
            except Exception as e:
                print(f"[mem0] 検索エラー: {e}")

        # ── RAG 検索 ──────────────────────────────────────────────────
        if use_rag and rag_manager and user_content:
            yield f"data: {json.dumps({'type': 'status', 'message': '📄 RAG検索中...'})}\n\n"
            hits = await rag_manager.search(user_content)
            if hits:
                context_text = "\n\n".join(h["content"] for h in hits)
                rag_system = {
                    "role": "system",
                    "content": (
                        "[STRICT INSTRUCTION]\n"
                        "以下に提供されたドキュメントコンテキストのみを使用して質問に答えてください。\n\n"
                        "ルール:\n"
                        "1. 以下のデータに明示されている事実のみを使用する。\n"
                        "2. 学習知識や事前情報は使用しない。\n"
                        "3. データに答えがない場合は「提供されたデータに記載がありません」と回答する。\n"
                        "4. 明示されていない関係性を推測・推論しない。\n\n"
                        "【ドキュメントコンテキスト】\n"
                        f"{context_text}"
                    ),
                }
                chat_messages = [rag_system] + chat_messages
                rag_sources = [{"source": h["source"], "score": h["score"]} for h in hits]
            yield f"data: {json.dumps({'type': 'rag_done', 'sources': rag_sources})}\n\n"

        # ── ストリーミングチャット ─────────────────────────────────────
        full_reply = ""
        async for chunk_json in chat_with_tools_streaming(
            messages=chat_messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            try:
                data = json.loads(chunk_json)
                if data["type"] == "chunk":
                    full_reply += data.get("content", "")
                elif data["type"] == "done":
                    full_reply = data.get("content", full_reply)
            except Exception:
                pass
            yield f"data: {chunk_json}\n\n"

        # ── 履歴保存・メタデータ送信 ──────────────────────────────────
        if user_content:
            append_to_history(session, "user", user_content)
        if full_reply:
            append_to_history(session, "assistant", full_reply)

        # ── メモリ保存（会話後） ────────────────────────────────────────
        # infer=False: LLM抽出をスキップして直接埋め込み保存（ローカルLLM互換）
        if use_memory and mem0_memory and user_content and full_reply:
            try:
                user_id = _get_user_id(session)
                result = await mem0_memory.add(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": full_reply},
                    ],
                    user_id=user_id,
                    infer=False,
                )
                added = result.get("results", result) if isinstance(result, dict) else result
                print(f"[mem0] メモリ保存: {len(added) if isinstance(added, list) else added}")
            except Exception as e:
                print(f"[mem0] 追加エラー: {e}")

        yield f"data: {json.dumps({'type': 'meta', 'history': get_session_history(session), 'rag_sources': rag_sources, 'memory_updated': use_memory})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── MCP ツール実行 ──────────────────────────────────────────────────────

async def call_mcp_tool(tool_name: str, tool_args: dict) -> str:
    try:
        async with stdio_client(config.Config.get_mcp_server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_args)
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                return "\n".join(parts)
    except Exception as e:
        return json.dumps({"error": f"MCPツール呼び出しエラー: {str(e)}"}, ensure_ascii=False)


async def chat_with_tools_streaming(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    max_tool_iterations: int = 5,
):
    """
    LM Studio とのチャットをストリーミングで行い、JSON 文字列を yield する。

    イベント種別:
      {"type": "chunk",     "content": "..."}   テキストチャンク（リアルタイム）
      {"type": "done",      "content": "..."}   最終テキスト（ツールなし完了時）
      {"type": "tool_call", "name": "..."}      ツール実行中
      {"type": "status",    "message": "..."}   状態メッセージ
      {"type": "error",     "content": "..."}   エラー

    テキスト応答はリアルタイムにストリームされる。
    ツール呼び出しがある場合は完了まで処理し、最終テキスト応答をストリームする。
    """
    current_messages = list(messages)
    tool_definitions = list(tools) if tools else []

    api_key = config.Config.get_api_key()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for iteration in range(max_tool_iterations):
        payload: dict = {
            "messages": current_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if model:
            payload["model"] = model
        if tool_definitions:
            payload["tools"] = tool_definitions
            payload["tool_choice"] = "auto"

        print(f"[chat_streaming] iteration={iteration}, messages={len(current_messages)}")

        full_content = ""
        tool_calls_acc: dict[int, dict] = {}
        has_tool_calls = False

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    config.Config.get_api_endpoint(),
                    headers=headers,
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})

                            # テキストチャンク → ツール呼び出しがなければリアルタイム送信
                            content = delta.get("content") or ""
                            if content and not has_tool_calls:
                                full_content += content
                                yield json.dumps({"type": "chunk", "content": content})

                            # ツールコールデルタを蓄積
                            for tc in delta.get("tool_calls") or []:
                                has_tool_calls = True
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {
                                        "id": "", "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if tc.get("id"):
                                    tool_calls_acc[idx]["id"] = tc["id"]
                                if fn := tc.get("function"):
                                    tool_calls_acc[idx]["function"]["name"]      += fn.get("name", "")
                                    tool_calls_acc[idx]["function"]["arguments"] += fn.get("arguments", "")
                        except Exception:
                            pass

        except httpx.HTTPStatusError as e:
            yield json.dumps({"type": "error", "content": f"⚠️ HTTPエラー: {e.response.status_code}"})
            return
        except Exception as e:
            yield json.dumps({"type": "error", "content": f"⚠️ API通信エラー: {str(e)}"})
            return

        tool_calls = list(tool_calls_acc.values()) if tool_calls_acc else []

        if not tool_calls:
            # テキスト応答で完了
            yield json.dumps({"type": "done", "content": full_content})
            return

        # ── ツール呼び出し処理 ──────────────────────────────────────
        current_messages.append({
            "role": "assistant",
            "content": full_content or None,
            "tool_calls": tool_calls,
        })

        for call in tool_calls:
            func_name = call["function"]["name"]
            raw_args  = call["function"].get("arguments", "{}")
            try:
                func_args = json.loads(raw_args)
            except json.JSONDecodeError:
                func_args = {}

            sanitized_args = {
                k: (None if v in (None, "None", "null", "") else v)
                for k, v in func_args.items()
            }

            yield json.dumps({"type": "tool_call", "name": func_name})
            print(f"[chat_streaming] MCP tool: {func_name}({sanitized_args})")
            result_content = await call_mcp_tool(func_name, sanitized_args)

            current_messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result_content,
            })

    yield json.dumps({"type": "error", "content": "⚠️ ツール呼び出しの最大反復回数を超えました。"})


# ─── mem0 メモリ API ─────────────────────────────────────────────────────


@app.get("/api/memory")
async def api_get_memory(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not mem0_memory:
        return JSONResponse({"memories": [], "available": False})
    try:
        user_id = _get_user_id(session)
        result = await mem0_memory.get_all(filters={"user_id": user_id}, top_k=50)
        memories = _parse_mem0_results(result)
        return JSONResponse({"memories": memories, "available": True})
    except Exception as e:
        return JSONResponse({"memories": [], "available": True, "error": str(e)})


@app.delete("/api/memory/{memory_id}")
async def api_delete_memory(memory_id: str, request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not mem0_memory:
        raise HTTPException(status_code=503, detail="mem0が初期化されていません")
    try:
        await mem0_memory.delete(memory_id)
        return JSONResponse({"status": "deleted"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/memory/clear")
async def api_clear_memory(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not mem0_memory:
        raise HTTPException(status_code=503, detail="mem0が初期化されていません")
    try:
        user_id = _get_user_id(session)
        await mem0_memory.delete_all(user_id=user_id)
        return JSONResponse({"status": "cleared"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/memory/status")
async def api_memory_status(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    return JSONResponse({"available": mem0_memory is not None})


@app.get("/api/history/{session_id}")
async def api_get_history(session_id: str, request: Request):
    _ = check_auth(request)
    return JSONResponse({"history": get_session_history(session_id)})


@app.post("/api/clear-history")
async def api_clear_history(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    clear_session_history(session)
    return JSONResponse({"status": "cleared"})


@app.get("/api/settings")
async def api_get_settings(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    return JSONResponse({
        "lm_studio_host": os.getenv("LM_STUDIO_HOST", "127.0.0.1"),
        "lm_studio_port": os.getenv("LM_STUDIO_PORT", "1234"),
        "has_api_key": bool(os.getenv("LM_STUDIO_API_KEY", "")),
        "app_username": os.getenv("APP_USERNAME", "admin"),
    })


@app.post("/api/update-settings")
async def api_update_settings(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    env_path = BASE_DIR / ".env"

    updates: dict[str, str] = {
        "LM_STUDIO_HOST": str(body.get("lm_studio_host", "")),
        "LM_STUDIO_PORT": str(body.get("lm_studio_port", "")),
        "LM_STUDIO_API_KEY": str(body.get("lm_studio_api_key", "")),
        "APP_USERNAME": str(body.get("app_username", "")),
    }
    if body.get("app_password"):
        updates["APP_PASSWORD"] = str(body["app_password"])

    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    for key, value in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    load_dotenv(env_path, override=True)

    return JSONResponse({"status": "updated", "message": "設定を更新しました。サーバーを再起動してください。"})


# ─── Function Calling ツール設定 ─────────────────────────────────────────

TOOLS_CONFIG_FILE = BASE_DIR / "tools_config.json"


@app.get("/api/tools")
async def api_get_tools(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    if TOOLS_CONFIG_FILE.exists():
        tools = json.loads(TOOLS_CONFIG_FILE.read_text(encoding="utf-8"))
    else:
        tools = []
    return JSONResponse({"tools": tools})


@app.post("/api/tools")
async def api_save_tools(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    tools = body.get("tools", [])
    TOOLS_CONFIG_FILE.write_text(
        json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return JSONResponse({"status": "saved", "count": len(tools)})


# ─── Corpus2Skill エンドポイント ─────────────────────────────────────────


@app.get("/api/rag/config")
async def rag_get_config(request: Request):
    """Corpus2Skill 設定を取得"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    cfg = load_rag_config()
    if rag_manager:
        status = rag_manager.get_status()
        cfg["llm_model"] = status.get("llm_model", "")
    return JSONResponse(cfg)


@app.post("/api/rag/config")
async def rag_save_config(request: Request):
    """Corpus2Skill 設定（LLM モデル・スキルツリーパラメータ）を保存"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    llm_model       = str(body.get("llm_model", "")).strip()
    max_top_skills  = int(body.get("max_top_skills", 6))
    branching_factor = int(body.get("branching_factor", 4))
    chunk_max_chars = int(body.get("chunk_max_chars", 800))

    RAG_CONFIG_FILE.write_text(
        json.dumps({
            "llm_model":        llm_model,
            "max_top_skills":   max_top_skills,
            "branching_factor": branching_factor,
            "chunk_max_chars":  chunk_max_chars,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if rag_manager:
        current_model = rag_manager.get_status().get("llm_model", "")
        if llm_model != current_model:
            await rag_manager.set_llm_model(llm_model)
        await rag_manager.set_compile_params(max_top_skills, branching_factor, chunk_max_chars)

    return JSONResponse({
        "status": "saved",
        "llm_model":        llm_model,
        "max_top_skills":   max_top_skills,
        "branching_factor": branching_factor,
        "chunk_max_chars":  chunk_max_chars,
    })


@app.get("/api/rag/status")
async def rag_status(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)
    return JSONResponse(rag_manager.get_status())


@app.post("/api/rag/upload")
async def rag_upload(request: Request, file: UploadFile = File(...)):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        return JSONResponse({"error": f"未対応の形式: {suffix}"}, status_code=400)

    dest = RAG_DOCS_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = await rag_manager.add_document(dest)
    return JSONResponse(result)


@app.post("/api/rag/index-dir")
async def rag_index_dir(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    results = await rag_manager.add_directory(RAG_DOCS_DIR)
    success = sum(1 for r in results if r.get("success"))
    return JSONResponse({
        "total": len(results),
        "success": success,
        "failed": len(results) - success,
        "details": results,
    })


@app.delete("/api/rag/document/{file_name}")
async def rag_delete_document(file_name: str, request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    result = await rag_manager.delete_document(file_name)
    return JSONResponse(result)


@app.delete("/api/rag/clear")
async def rag_clear(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    result = await rag_manager.clear()
    return JSONResponse(result)


@app.post("/api/rag/search")
async def rag_search(request: Request):
    """RAG 検索テスト用エンドポイント（detail=true で個別チャンク返却）"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    body = await request.json()
    query  = body.get("query", "")
    detail = bool(body.get("detail", False))
    top_k  = int(body.get("top_k", 10))
    if not query:
        return JSONResponse({"error": "queryが空です"}, status_code=400)

    if detail:
        chunks = await rag_manager.search_chunks(query, top_k=top_k)
        return JSONResponse({"chunks": chunks})
    else:
        hits = await rag_manager.search(query)
        return JSONResponse({"results": hits})


@app.get("/api/rag/compile-status")
async def rag_compile_status_endpoint(request: Request):
    """コンパイル進捗状態を返す"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"state": "error", "current_skill": 0, "total_skills": 0, "message": "RAGが初期化されていません"})
    return JSONResponse(rag_manager.get_compile_status())


@app.post("/api/rag/recompile")
async def rag_recompile(request: Request):
    """現在のドキュメントでスキルツリーを再構築する"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not rag_manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)
    result = await rag_manager.recompile()
    return JSONResponse(result)


# ─── スキルツリービューア ──────────────────────────────────────────────────

@app.get("/rag-search", response_class=HTMLResponse)
async def rag_search_page(request: Request):
    """RAG 検索テストページ"""
    session = request.cookies.get("session_token")
    if not session:
        return HTMLResponse('<script>window.location.href="/";</script>')
    return templates.TemplateResponse(name="rag_search.html", context={}, request=request)


@app.get("/skill-tree", response_class=HTMLResponse)
async def skill_tree_page(request: Request):
    """スキルツリービューア ページ"""
    session = request.cookies.get("session_token")
    if not session:
        return HTMLResponse('<script>window.location.href="/";</script>')
    return templates.TemplateResponse(name="skill_tree.html", context={}, request=request)


@app.get("/api/skill-tree")
async def api_skill_tree(request: Request):
    """スキルツリー構造 + チャンクテキストを返す"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    working_dir = BASE_DIR / "c2s_db"
    skill_meta_path  = working_dir / "skill_meta.json"
    chunk_index_path = working_dir / "chunk_index.json"

    if not skill_meta_path.exists():
        return JSONResponse({"skills": [], "total_chunks": 0})

    skills = json.loads(skill_meta_path.read_text(encoding="utf-8"))

    # チャンクテキストを (doc_id, chunk_idx) → text の辞書に
    chunk_map: dict[tuple[str, int], str] = {}
    if chunk_index_path.exists():
        for entry in json.loads(chunk_index_path.read_text(encoding="utf-8")):
            chunk_map[(entry["doc_id"], entry["chunk_idx"])] = entry["text"]

    # 各 sub_skill に chunk テキストを付加
    for skill in skills:
        for sub in skill.get("sub_skills", []):
            sub_dir = working_dir / sub["dir"]
            ids_path = sub_dir / "chunk_ids.json"
            sub["chunks"] = []
            if ids_path.exists():
                for ref in json.loads(ids_path.read_text(encoding="utf-8")):
                    text = chunk_map.get((ref["doc_id"], ref["chunk_idx"]), "")
                    sub["chunks"].append({
                        "doc_id":    ref["doc_id"],
                        "chunk_idx": ref["chunk_idx"],
                        "text":      text,
                    })

    total_chunks = sum(
        len(sub.get("chunks", []))
        for skill in skills
        for sub in skill.get("sub_skills", [])
    )

    return JSONResponse({"skills": skills, "total_chunks": total_chunks})


# ─── メイン ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.Config.get_app_host(),
        port=config.Config.get_app_port(),
        reload=True,
    )
