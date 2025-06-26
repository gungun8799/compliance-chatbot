# Run application locally using this command: chainlit run app.py -h --root-path /chatbot/v1
import asyncio
import json
import logging
import os
import re
import time
import uuid
import warnings
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import chainlit as cl
import httpx
import markdown
import redis
from bs4 import BeautifulSoup
from chainlit import Action
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.llms.groq import Groq
from llama_index.llms.openai_like import OpenAILike
from llama_index.storage.chat_store.redis import RedisChatStore
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
from sqlalchemy import JSON, Column, MetaData, String, Table, select
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from functions.qdrant_vectordb import QdrantManager
# Apply the monkey patch
from patches import patch
from prompts import SYSTEM_PROMPT_DEEPTHINK, SYSTEM_PROMPT_STANDARD

patch.apply_patch()

# ======================================================================================
# Configuration and Initialization
# ======================================================================================

# Load from root .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Environment Variables
GROQ_MODEL_ID_1 = os.getenv("GROQ_MODEL_ID_1")
GROQ_MODEL_ID_2 = os.getenv("GROQ_MODEL_ID_2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")
API_KEY_CHATBOT = os.getenv("API_KEY_CHATBOT")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID")
COHERE_MODEL_ID = os.getenv("COHERE_MODEL_ID")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
REDIS_CHATSTORE_URI = os.getenv("REDIS_CHATSTORE_URI")
REDIS_CHATSTORE_PASSWORD = os.getenv("REDIS_CHATSTORE_PASSWORD")
TOKEN_LIMIT = 512
TRACE_ENDPOINT = os.getenv("TRACE_ENDPOINT")
TRACE_PROJECT_NAME = os.getenv("TRACE_PROJECT_NAME")
MS_TEAMS_WORKFLOW_URL = os.getenv("MS_TEAMS_WORKFLOW_URL")
CHAINLIT_AUTH_SECRET = os.getenv("CHAINLIT_AUTH_SECRET")

print("📡 MS_TEAMS_WORKFLOW_URL:", MS_TEAMS_WORKFLOW_URL)
print("✅ Loaded CHAINLIT_AUTH_SECRET:", CHAINLIT_AUTH_SECRET)


# Constants
MAX_CLARIFICATION_ROUNDS = 2
MAX_TOPICS_BEFORE_CLARIFY = 3
SIMILARITY_TIE_THRESHOLD = 0.03
FUZZY_THRESHOLD = 0.7
VECTOR_MIN_THRESHOLD = 0.45
VECTOR_MEDIUM_THRESHOLD = 0.75

# Redis Client
parsed_redis_url = urlparse(REDIS_CHATSTORE_URI)
redis_client = redis.Redis(
    host=parsed_redis_url.hostname,
    port=parsed_redis_url.port or 6379,
    password=REDIS_CHATSTORE_PASSWORD,
    db=0,
)

# Phoenix Tracer
tracer_provider = register(
    project_name=TRACE_PROJECT_NAME,
    endpoint=TRACE_ENDPOINT,
    set_global_tracer_provider=False,
)
LlamaIndexInstrumentor().instrument(
    skip_dep_check=True, tracer_provider=tracer_provider
)

# SQLAlchemy Metadata
extra_meta = MetaData()
clarification_state = Table(
    "clarification_state",
    extra_meta,
    Column("thread_id", String, primary_key=True),
    Column("summaries", JSON, nullable=False),
    Column("nodes", JSON, nullable=False),
)

# Chat Store
chat_store = RedisChatStore(
    redis_url=REDIS_CHATSTORE_URI, db=0, password=REDIS_CHATSTORE_PASSWORD, ttl=180
)

# Qdrant Manager
qdrant_manager = QdrantManager()

# LlamaIndex Settings
Settings.context_window = 12000

# Dynamically set embedding model based on .env file
EMBEDDING_SERVICE = os.getenv("EMBEDDING_SERVICE", "text_embeddings_inference").lower()

if EMBEDDING_SERVICE == "cohere":
    logger.info(f"Using Cohere for embeddings. Model: {os.getenv('COHERE_MODEL_ID')}")
    Settings.embed_model = CohereEmbedding(
        api_key=os.getenv("COHERE_API_KEY"),
        model_name=os.getenv("COHERE_MODEL_ID"),
        input_type="search_document",
        embedding_type="float",
    )
else:
    logger.info(f"Using TextEmbeddingsInference for embeddings. Model: {EMBED_MODEL_ID}")
    Settings.embed_model = TextEmbeddingsInference(
        model_name=EMBED_MODEL_ID,
        base_url=EMBED_BASE_URL,
        auth_token=f"Bearer {API_KEY_CHATBOT}",
        timeout=60,
        embed_batch_size=10,
    )

# Chat Profiles
DATASET_MAPPING = {
    "Standard": QDRANT_COLLECTION_NAME,
    "Deepthink": QDRANT_COLLECTION_NAME,
    "Accounting Compliance": QDRANT_COLLECTION_NAME,
}

CHAT_PROFILES = {
    "Deepthink": {
        "context_prompt": SYSTEM_PROMPT_DEEPTHINK,
        "welcome_message": "Hello {firstname}, how can I help you today?",
        "llm_settings": {
            "model": GROQ_MODEL_ID_2,
            "api_key": GROQ_API_KEY,
            "is_chat_model": True,
            "is_function_calling_model": False,
            "temperature": 0.7,
        },
    },
    "Accounting Compliance": {
        "context_prompt": SYSTEM_PROMPT_STANDARD,
        "welcome_message": "Hi there! Need help with accounting compliance?",
        "llm_settings": {
            "model": LLM_MODEL_ID,
            "api_base": LLM_BASE_URL,
            "api_key": API_KEY_CHATBOT,
            "is_chat_model": True,
            "is_function_calling_model": False,
            "temperature": 0.7,
            "http_client": httpx.Client(verify=False),
        },
    },
}

# Load predefined answers
with open("predefined_answers.json", "r", encoding="utf-8") as f:
    predefined_answers = json.load(f)

# Global state trackers
shown_admin_replies = {}
shown_admin_reply_ids = {}
shown_parent_keys = set()


# ======================================================================================
# Utility Functions
# ======================================================================================

def strip_html(html: str) -> str:
    """Removes HTML tags from a string."""
    return re.sub("<[^<]+?>", "", html).strip()


def extract_and_format_table(text: str) -> str:
    """
    Detects and reformats Markdown tables into neat, aligned tables.
    Non-table text is left untouched.
    """
    lines = text.splitlines()
    output_lines = []
    buffer = []

    def flush_table():
        nonlocal buffer, output_lines
        rows = [
            re.split(r"\s*\|\s*", row.strip("| "))
            for row in buffer
            if row.strip() and not re.fullmatch(r"[\|\-\s]+", row)
        ]
        if not rows:
            buffer = []
            return

        max_cols = max(len(r) for r in rows)
        for r in rows:
            r.extend([""] * (max_cols - len(r)))

        widths = [max(len(r[i]) for r in rows) for i in range(max_cols)]
        header = "| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(max_cols)) + " |"
        sep = "|" + "|".join("-" * (widths[i] + 2) for i in range(max_cols)) + "|"
        output_lines.extend([header, sep])

        for row in rows[1:]:
            line = "| " + " | ".join(row[i].ljust(widths[i]) for i in range(max_cols)) + " |"
            output_lines.append(line)
        buffer = []

    for line in lines:
        if "|" in line or re.fullmatch(r"[\|\-\s]+", line):
            buffer.append(line)
        else:
            if buffer:
                flush_table()
            output_lines.append(line)

    if buffer:
        flush_table()

    return "\n".join(output_lines)


def clean_parent_content(raw_html: str) -> str:
    """Strips HTML and removes metadata lines from content."""
    text = strip_html(raw_html)
    lines = text.splitlines()
    return "\n".join(
        line for line in lines if not any(tag in line for tag in ["[thread_id:", "[parent_id:", "Email:"])
    ).strip()


def save_conversation_log(thread_id: str, parent_id: str, role: str, content: str, difficulty: str = None):
    """Saves a conversation log entry to Redis."""
    key = f"conversation_log:{thread_id}"
    log_entry = {"timestamp": time.time(), "parent_id": parent_id, "role": role, "content": content}
    if difficulty:
        log_entry["difficulty"] = difficulty

    existing_raw = redis_client.get(key)
    log_list = json.loads(existing_raw) if existing_raw else []
    log_list.append(log_entry)
    redis_client.set(key, json.dumps(log_list))
    logger.info(f"📝 Logged {role} message to {key}")


async def send_with_feedback(content: str, author: str = "Customer Service Agent", parent_id: str = None, metadata: Optional[Dict] = None):
    """Sends a message and streams it character by character."""
    msg = cl.Message(content="", author=author, parent_id=parent_id, metadata=metadata or {})
    await msg.send()
    for char in content:
        await msg.stream_token(char)
        await asyncio.sleep(0.005)
    await msg.update()


# ======================================================================================
# LLM and Chat Engine Setup
# ======================================================================================

@cl.data_layer
def get_data_layer():
    """Returns the SQLAlchemy data layer."""
    return SQLAlchemyDataLayer(conninfo=os.environ["ASYNC_DATABASE_URL"])


def get_llm_settings(chat_profile: str):
    """Retrieves and configures LLM settings for a given chat profile."""
    settings = CHAT_PROFILES.get(chat_profile, {}).get("llm_settings")
    if not settings:
        raise ValueError(f"No LLM settings found for profile: {chat_profile}")

    if chat_profile == "Accounting Compliance":
        return OpenAILike(**settings)
    elif chat_profile == "Deepthink":
        return Groq(**settings)
    else:
        raise ValueError(f"Unsupported chat profile: {chat_profile}")


def create_chat_engine(chat_profile: str):
    """Creates a chat engine and retriever for a given profile."""
    dataset = DATASET_MAPPING.get(chat_profile)
    if not dataset:
        logger.error(f"No dataset configured for profile: {chat_profile}")
        return None, None

    vector_store = qdrant_manager.get_vector_store(dataset, hybrid=True)
    if not vector_store:
        logger.error(f"❌ Failed to get vector store for dataset: {dataset}")
        return None, None

    index = VectorStoreIndex.from_vector_store(vector_store)
    llm = get_llm_settings(chat_profile)

    query_engine = index.as_query_engine(
        retriever_mode="hybrid", llm=llm, streaming=True, verbose=True,
        similarity_top_k=5, sparse_top_k=12, alpha=0.5
    )
    retriever = index.as_retriever(similarity_top_k=5)
    return query_engine, retriever


def setup_runnable():
    """Sets up the runnable (chat engine) and retriever in the user session."""
    try:
        chat_profile = cl.user_session.get("chat_profile")
        if not chat_profile:
            logger.error("chat_profile not found in user session.")
            return

        Settings.llm = get_llm_settings(chat_profile)
        chat_engine, retriever = create_chat_engine(chat_profile)
        if chat_engine and retriever:
            cl.user_session.set("runnable", chat_engine)
            cl.user_session.set("retriever", retriever)
        else:
            logger.warning("Failed to create chat engine or retriever.")
    except Exception as e:
        logger.exception("Error setting up runnable: %s", e)


# ======================================================================================
# Clarification Flow Logic
# ======================================================================================

def clear_clarification_state():
    """Clears all clarification-related state from the user session."""
    for key in [
        "awaiting_clarification", "clarification_rounds", "possible_summaries",
        "nodes_to_consider", "summary_to_meta", "original_query"
    ]:
        cl.user_session.set(key, None)


async def answer_from_node(node, user_q):
    """Builds and sends the final LLM response from a single selected node."""
    src = node.node.metadata.get("source", "Unknown")
    txt = node.node.text.replace("\n", " ")
    prompt = (
        f'ผู้ใช้ถามว่า: "{user_q}"\n'
        f'บทความที่เลือกมาจากเอกสารนโยบาย "{src}" (เต็มข้อความ):\n"""{txt}\n"""\n\n'
        "กรุณาตอบโดยอาศัยเนื้อหาในบทความนี้"
    )
    runnable = cl.user_session.get("runnable")
    resp = runnable.query(prompt)
    answer = resp.response if hasattr(resp, "response") else "".join(resp.response_gen)
    answer = extract_and_format_table(answer.strip())

    clear_clarification_state()

    msg = cl.Message(content="", metadata={"difficulty": "Clarified"})
    await msg.send()
    stream_text = f"✅ นี่คือสิ่งที่พบจาก “{src}”:\n\n{answer}"
    for char in stream_text:
        await msg.stream_token(char)
        await asyncio.sleep(0.005)
    await msg.update()

    save_conversation_log(cl.context.session.thread_id, None, "bot", answer, difficulty="Clarified")


# ======================================================================================
# Chainlit Event Handlers
# ======================================================================================

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Handles user authentication."""
    if (username, password) == ("admin", "admin"):
        logger.info("✅ Login success for admin")
        return cl.User(identifier="admin", metadata={"role": "ADMIN", "email": "chatbot_admin@gmail.com", "provider": "credentials"})
    logger.warning("❌ Login failed")
    return None


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    """Sets the available chat profiles."""
    return [
        cl.ChatProfile(
            name="Accounting Compliance",
            markdown_description="Got questions about the policy? I'm all ears and ready to help you out—just ask!",
            icon="/public/cp_accountant.png",
        ),
    ]


@cl.set_starters
async def set_starters():
    """Sets the starter questions for the welcome screen."""
    return [
        cl.Starter(label="อำนาจอนุมัติการลงทุน investment project แต่ละประเภท แต่ละมูลค่า", message="อำนาจอนุมัติการลงทุน investment project แต่ละประเภท แต่ละมูลค่า", icon="/public/search.svg"),
        cl.Starter(label="เอกสารที่ต้องใช้สำหรับการเปิด new vendor code มีอะไรบ้าง ?", message="เอกสารที่ต้องใช้สำหรับการเปิด new vendor code มีอะไรบ้าง ?", icon="/public/search.svg"),
        cl.Starter(label="รอบการทำเบิกเงินทดรองจ่าย และการจ่ายเงิน", message="รอบการทำเบิกเงินทดรองจ่าย และการจ่ายเงิน", icon="/public/search.svg"),
        cl.Starter(label="เมื่อไรต้องเปิด PR ผ่านระบบ เมื่อไรสามารถใช้ PO manual (PO กระดาษได้)", message="เมื่อไรต้องเปิด PR ผ่านระบบ เมื่อไรสามารถใช้ PO manual (PO กระดาษได้)", icon="/public/search.svg"),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Initializes the chat session."""
    logger.info(f"💬 on_chat_start called for user: {cl.user_session.get('user')}")
    thread_id = cl.context.session.thread_id
    dl: SQLAlchemyDataLayer = get_data_layer()
    engine = dl.engine

    # Persist thread row
    meta = MetaData()
    threads_table = Table("threads", meta, Column("id", PG_UUID(as_uuid=True), primary_key=True))
    thread_uuid = uuid.UUID(thread_id)
    async with engine.begin() as conn:
        await conn.execute(pg_insert(threads_table).values(id=thread_uuid).on_conflict_do_nothing())

    # Setup memory and runnable
    app_user = cl.user_session.get("user")
    redis_session_id = f"{app_user.identifier}:{thread_id}"
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=TOKEN_LIMIT, chat_store=chat_store, chat_store_key=redis_session_id
    )
    cl.user_session.set("memory", memory)
    setup_runnable()

    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile:
        llm_model = CHAT_PROFILES.get(chat_profile, {}).get("llm_settings", {}).get("model")
        logger.info(f"Chat started with profile: '{chat_profile}', LLM Model ID: '{llm_model}'")

    logger.info(f"🚀 Starting poll_all_admin_replies for thread: {thread_id}")
    asyncio.create_task(poll_all_admin_replies(thread_id))


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Handles chat resumption from history."""
    thread_id = thread.get("id")
    app_user = cl.user_session.get("user")
    redis_session_id = f"{app_user.identifier}:{thread_id}"

    # Rebuild memory
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=TOKEN_LIMIT, chat_store=chat_store, chat_store_key=redis_session_id
    )
    root_messages = [m for m in thread["steps"] if m["parentId"] is None and m.get("output", "").strip()]
    for message in root_messages:
        role = "user" if message["type"] == "user_message" else "assistant"
        memory.put(ChatMessage(role=role, content=message["output"]))
    cl.user_session.set("memory", memory)

    # Load saved clarification state
    dl = get_data_layer()
    engine = dl.engine
    async with AsyncSession(engine) as session:
        result = await session.execute(select(clarification_state).where(clarification_state.c.thread_id == thread_id))
        row = result.mappings().first()

    if row:
        data = dict(row)
        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("possible_summaries", data["summaries"])
        from types import SimpleNamespace
        nodes = [SimpleNamespace(score=n["score"], node=SimpleNamespace(text=n["text"], metadata=n["meta"])) for n in data["nodes"]]
        cl.user_session.set("nodes_to_consider", nodes)

    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    """Handles incoming user messages."""
    logger.info(f"Received message from user: {message.content}")
    runnable = cl.user_session.get("runnable")
    retriever = cl.user_session.get("retriever")
    thread_id = cl.context.session.thread_id

    save_conversation_log(thread_id, message.id, role="user", content=message.content)

    if not runnable or not retriever:
        await send_with_feedback("⚠️ System not ready. Please try again later.")
        return

    if cl.user_session.get("awaiting_clarification"):
        await handle_clarification_response(message)
    else:
        await handle_standard_query(message)


# ======================================================================================
# Message Handling Logic
# ======================================================================================

async def handle_clarification_response(message: cl.Message):
    """Handles user's response during a clarification flow."""
    nodes_to_consider = cl.user_session.get("nodes_to_consider", [])
    summaries = cl.user_session.get("possible_summaries", [])
    original_query = cl.user_session.get("original_query", "")
    rounds = cl.user_session.get("clarification_rounds", 0)

    if rounds >= MAX_CLARIFICATION_ROUNDS:
        chosen = max(nodes_to_consider, key=lambda n: n.score)
        logger.info(f"[clarify] Max rounds reached, auto-selecting node with score {chosen.score:.2f}")
        clear_clarification_state()
        return await answer_from_node(chosen, original_query)

    cl.user_session.set("clarification_rounds", rounds + 1)
    choice = message.content.strip()

    if choice.lower() in ["❌", "❌ ถามคำถามใหม่", "exit", "ถามคำถามใหม่"]:
        clear_clarification_state()
        await send_with_feedback("✅ คุณได้เลือกเริ่มต้นคำถามใหม่ กรุณาถามคำถามของคุณอีกครั้ง")
        return

    if choice.lower() == "auto":
        chosen = max(nodes_to_consider, key=lambda n: n.score)
        logger.info(f"[clarify] User requested auto, selecting node with score {chosen.score:.2f}")
        clear_clarification_state()
        return await answer_from_node(chosen, original_query)

    # Determine selected index
    selected_index = None
    if choice.isdigit() and 0 <= int(choice) - 1 < len(summaries):
        selected_index = int(choice) - 1
    else:
        ratios = [SequenceMatcher(None, choice.lower(), s.lower()).ratio() for s in summaries]
        if max(ratios) > 0.6:
            selected_index = ratios.index(max(ratios))

    if selected_index is None:
        await send_with_feedback("⚠️ ไม่พบหัวข้อที่เลือก โปรดพิมพ์หมายเลขหรือชื่อหัวข้อให้ถูกต้องอีกครั้ง")
        return

    # Handle tie-breaking
    chosen_node = nodes_to_consider[selected_index]
    tied_nodes = [n for n in nodes_to_consider if abs(n.score - chosen_node.score) < SIMILARITY_TIE_THRESHOLD]

    if len(tied_nodes) > 1:
        await re_clarify(tied_nodes, original_query)
    else:
        clear_clarification_state()
        await answer_from_node(chosen_node, original_query)


async def re_clarify(nodes: list, original_query: str):
    """Asks for another round of clarification on a smaller set of nodes."""
    llm = get_llm_settings(cl.user_session.get("chat_profile"))
    new_summaries, new_meta = [], {}

    for n in nodes:
        trunc = n.node.text[:1000] + "…" if len(n.node.text) > 1000 else n.node.text
        prompt = (
            f'ผู้ใช้ถามว่า: "{original_query}"\n'
            f'เนื้อหาในเอกสารนี้ (เต็มข้อความ):\n"""{trunc}\n"""\n'
            "กรุณาสรุปเนื้อหานี้เป็นหัวข้อสั้น ๆ (ไม่เกิน 10 คำ) เพื่อให้ผู้ใช้เลือกหัวข้ออีกครั้ง"
        )
        resp = llm.chat([ChatMessage(role="user", content=prompt)])
        one_line = resp.message.content.strip().split("\n")[0]
        if one_line not in new_meta:
            new_meta[one_line] = n
            new_summaries.append(one_line)

    opt_out_choice = "❌ ถามคำถามใหม่"
    if opt_out_choice not in new_summaries:
        new_summaries.append(opt_out_choice)

    cl.user_session.set("nodes_to_consider", nodes)
    cl.user_session.set("possible_summaries", new_summaries)
    cl.user_session.set("summary_to_meta", new_meta)

    await send_with_feedback(
        content=(
            "❓ ยังพบเนื้อหาหลายรายการที่คะแนนใกล้เคียงกัน กรุณาช่วยระบุหัวข้อเพิ่มเติม\n\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(new_summaries))
            + '\n\nโปรดตอบกลับด้วยหมายเลขหรือชื่อหัวข้อที่ต้องการ หรือเลือก "❌ ถามคำถามใหม่" หากต้องการเริ่มต้นใหม่'
        ),
        author="Customer Service Agent",
    )


async def handle_standard_query(message: cl.Message):
    """Handles a standard, non-clarification query."""
    retriever = cl.user_session.get("retriever")
    runnable = cl.user_session.get("runnable")

    # Fuzzy match against predefined answers
    fuzzy_scores = {q: SequenceMatcher(None, message.content.lower(), q.lower()).ratio() for q in predefined_answers}
    best_question = max(fuzzy_scores, key=fuzzy_scores.get)
    fuzzy_score = fuzzy_scores[best_question]
    best_match = predefined_answers[best_question]

    # Vector retrieval
    try:
        nodes = retriever.retrieve(message.content)
        top_score = nodes[0].score if nodes else 0.0
    except Exception as e:
        await send_with_feedback(f"⚠️ Retrieval error: {str(e)}")
        return

    # Ambiguity check
    scores = [n.score for n in nodes[:MAX_TOPICS_BEFORE_CLARIFY]]
    if len(scores) >= 2 and all(abs(s - scores[0]) < SIMILARITY_TIE_THRESHOLD for s in scores) and scores[0] >= VECTOR_MIN_THRESHOLD:
        return await start_clarification_flow(nodes, message.content)

    # Determine response level
    if top_score >= VECTOR_MEDIUM_THRESHOLD:
        level = "Hard"
    elif top_score >= VECTOR_MIN_THRESHOLD:
        level = "Medium"
    else:
        level = "Rejected"

    # Generate response based on level
    if level == "Rejected":
        content = (
            "❌ คำถามนี้ไม่เกี่ยวข้องกับนโยบาย LOA / DoA กรุณาถามในหัวข้อที่เกี่ยวข้อง\n\n"
            f"🧠 *DEBUG* | Category: **Rejected** | Vector: {top_score:.2f} | Fuzzy: {fuzzy_score:.2f}"
        )
        await send_with_feedback(content, metadata={"difficulty": "Rejected"})
        save_conversation_log(cl.context.session.thread_id, message.id, "bot", "Rejected", "Reject")

    elif level == "Easy" and fuzzy_score > FUZZY_THRESHOLD:
        content = (
            f"{best_match}\n\n"
            f"🧠 *DEBUG* | Category: **Easy** | Method: **Predefined** | Vector: {top_score:.2f} | Fuzzy: {fuzzy_score:.2f} | Matched Q: {best_question}"
        )
        await send_with_feedback(content, metadata={"difficulty": "Easy"})
        save_conversation_log(cl.context.session.thread_id, message.id, "bot", best_match, "Easy")

    elif level == "Medium" or level == "Hard":
        await answer_with_llm(nodes, message.content, level, top_score, fuzzy_score)


async def start_clarification_flow(nodes: list, original_query: str):
    """Initiates the clarification process when a query is too broad."""
    llm = get_llm_settings(cl.user_session.get("chat_profile"))
    summaries, summary_to_meta = [], {}

    nodes_to_summarize = [n for n in nodes[:MAX_TOPICS_BEFORE_CLARIFY] if any(tok in n.node.text for tok in re.findall(r"\w+", original_query))]
    if len(nodes_to_summarize) < 2:
        nodes_to_summarize = nodes[:MAX_TOPICS_BEFORE_CLARIFY]

    for n in nodes_to_summarize:
        trunc = n.node.text[:1000] + "…" if len(n.node.text) > 1000 else n.node.text
        prompt = (
            f'ผู้ใช้ถามว่า: "{original_query}"\n'
            f'เนื้อหาในเอกสารนี้ (เต็มข้อความ):\n"""{trunc}\n"""\n'
            "กรุณาสรุปเนื้อหานี้เป็นหัวข้อเพื่อให้ผู้ใช้เลือก นี่คือหัวข้อย่อยที่ผู้ใช้จะเลือกเมื่อคำตอบอาจเป็นไปได้หลายกรณี เริ่มต้นประโยคด้วย กรณี เสมอ"
        )
        resp = llm.chat([ChatMessage(role="user", content=prompt)])
        one_line = resp.message.content.strip().split("\n")[0]
        if one_line not in summary_to_meta:
            summaries.append(one_line)
            summary_to_meta[one_line] = (n, trunc, n.node.metadata.get("source", "UnknownPolicy"))

    opt_out_choice = "❌ ถามคำถามใหม่"
    if opt_out_choice not in summaries:
        summaries.append(opt_out_choice)

    # Set session state for clarification
    cl.user_session.set("awaiting_clarification", True)
    cl.user_session.set("possible_summaries", summaries)
    cl.user_session.set("nodes_to_consider", nodes_to_summarize)
    cl.user_session.set("summary_to_meta", summary_to_meta)
    cl.user_session.set("original_query", original_query)

    # Persist state to DB
    payload = {
        "summaries": summaries,
        "nodes": [{"score": n.score, "text": n.node.text, "meta": n.node.metadata} for n in nodes_to_summarize],
    }
    dl = get_data_layer()
    engine = dl.engine
    async with AsyncSession(engine) as session:
        await session.execute(
            pg_insert(clarification_state).values(thread_id=cl.context.session.thread_id, **payload)
            .on_conflict_do_update(index_elements=["thread_id"], set_=payload)
        )
        await session.commit()

    await send_with_feedback(
        content=(
            "❓ พบเอกสารหลายรายการที่อาจเกี่ยวข้องกับคำถามของคุณ\n\n"
            "หัวข้อที่เป็นไปได้:\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
            + '\n\nโปรดตอบกลับด้วยหมายเลขหรือชื่อหัวข้อที่ต้องการ หรือเลือก "❌ ถามคำถามใหม่" หากต้องการเริ่มต้นใหม่'
        ),
        author="Customer Service Agent",
    )


async def answer_with_llm(nodes: list, query: str, level: str, top_score: float, fuzzy_score: float):
    """Generates an answer using the LLM with context from retrieved nodes."""
    runnable = cl.user_session.get("runnable")
    TOP_K = 3
    selected_nodes = nodes[:TOP_K]
    contexts = [(n.node.metadata.get("source", "Unknown"), n.node.text.strip().replace("\n", " ")) for n in selected_nodes]
    context_str = "".join(f'({i}) เอกสารนโยบาย: "{src}"\nเนื้อหาชิ้นนี้ (เต็มข้อความ):\n"""{txt}\n"""\n\n' for i, (src, txt) in enumerate(contexts, 1))

    filtered_query = (
        f'ผู้ใช้ถามว่า: "{query}"\n\n'
        f"กรุณาตอบโดยอาศัยเนื้อหาต่อไปนี้ทั้งหมด:\n\n{context_str}"
        "ในคำตอบให้ระบุชื่อเอกสารนโยบายที่ใช้ และชี้ว่าอ้างอิงมาจากข้อความใดบ้าง"
    )

    try:
        resp = runnable.query(filtered_query)
        answer_body = resp.response.strip() if hasattr(resp, "response") else "".join(resp.response_gen).strip()
    except Exception as e:
        answer_body = f"⚠️ LLM error: {e}"

    answer_body = extract_and_format_table(answer_body)
    sources_used = ", ".join(f'"{src}"' for src, _ in contexts)
    final_answer = (
        f"📚 จากเอกสารนโยบาย: {sources_used}\n\n{answer_body}\n\n"
        f"🧠 *DEBUG* | Category: **{level}** | Method: **VectorStore + LLM (top {TOP_K})** | Vector: {top_score:.2f} | Fuzzy: {fuzzy_score:.2f}"
    )

    await send_with_feedback(final_answer, metadata={"difficulty": level})
    save_conversation_log(cl.context.session.thread_id, cl.context.session.id, "bot", final_answer, level)


# ======================================================================================
# Background Tasks (Admin Replies)
# ======================================================================================

async def poll_all_admin_replies(thread_id: str):
    """Polls Redis for all admin replies in a given thread."""
    printed_keys = set()
    while True:
        try:
            keys = redis_client.keys(f"admin-reply:{thread_id}:*")
            for key in keys:
                key_str = key.decode("utf-8")
                raw = redis_client.get(key)
                if not raw:
                    continue

                payload = json.loads(raw.decode("utf-8"))
                parent_content = payload.get("parent_content", "")
                replies = payload.get("replies", [])
                parent_id = key_str.split(":")[2]
                last_reply_id = replies[-1]["id"] if replies else None

                if shown_admin_replies.get(key_str) == last_reply_id:
                    continue

                if key_str not in printed_keys:
                    await send_with_feedback(f"🧾 Original Question:\n\n{clean_parent_content(parent_content)}", author="User")
                    printed_keys.add(key_str)

                for r in replies:
                    reply_id = r.get("id")
                    if reply_id and not shown_admin_replies.get(f"{key_str}:{reply_id}"):
                        content = r.get("body", {}).get("content", "")
                        cleaned = strip_html(content)
                        if cleaned:
                            await send_with_feedback(f"📬 Reply from Admin:\n\n{cleaned}", author="Admin", parent_id=parent_id)
                            shown_admin_replies[f"{key_str}:{reply_id}"] = True

                if last_reply_id:
                    shown_admin_replies[key_str] = last_reply_id
        except Exception as e:
            logger.error(f"❌ Redis polling error: {e}")
        await asyncio.sleep(5)