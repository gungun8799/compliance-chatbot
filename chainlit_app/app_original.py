# Run application locally using this command: chainlit run app.py -h --root-path /chatbot/v1
import asyncio
import json
import logging
import os
import re
import time
import uuid
import warnings
import contextlib
from contextlib import suppress
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse
from typing import List


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
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from functions.qdrant_vectordb import QdrantManager
# Apply the monkey patch
from patches import patch
from prompts import SYSTEM_PROMPT_DEEPTHINK, SYSTEM_PROMPT_STANDARD

patch.apply_patch()

# ======================================================================================
# Configuration and Initialization
# ======================================================================================

# Determine environment mode
env_mode = os.getenv("ENV_MODE", "dev")  # default to "dev" if not set

# Build path to appropriate .env file
env_file = Path(__file__).resolve().parents[1] / f".env.{env_mode}"

# Load the selected .env file
load_dotenv(dotenv_path=env_file)

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
API_KEY_CHATBOT_PRI = os.getenv("API_KEY_CHATBOT_PRI")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID")
COHERE_MODEL_ID = os.getenv("COHERE_MODEL_ID")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
REDIS_CHATSTORE_URI = os.getenv("REDIS_CHATSTORE_URI")
REDIS_CHATSTORE_PASSWORD = os.getenv("REDIS_CHATSTORE_PASSWORD")
TOKEN_LIMIT = 512 # Default token limit for chat memory
TRACE_ENDPOINT = os.getenv("TRACE_ENDPOINT")
TRACE_PROJECT_NAME = os.getenv("TRACE_PROJECT_NAME")
MS_TEAMS_WORKFLOW_URL = os.getenv("MS_TEAMS_WORKFLOW_URL")
CHAINLIT_AUTH_SECRET = os.getenv("CHAINLIT_AUTH_SECRET")

logger.info(f"📡 MS_TEAMS_WORKFLOW_URL: {MS_TEAMS_WORKFLOW_URL}")
logger.info(f"✅ Loaded CHAINLIT_AUTH_SECRET: {CHAINLIT_AUTH_SECRET}")


# Constants
MAX_CLARIFICATION_ROUNDS = 2
MAX_FUZZY_CLARIFICATION_ROUNDS = 3
MAX_TOPICS_BEFORE_CLARIFY = 7
MAX_FUZZY_CLARIFY_TOPICS = 5
SIMILARITY_TIE_THRESHOLD = 0.03
FUZZY_THRESHOLD = 0.55
FUZZY_CLARIFY_THRESHOLD = 0.85  # 👈 triggers clarification when multiple fuzzy candidates exist
VECTOR_MIN_THRESHOLD = 0.3
VECTOR_MEDIUM_THRESHOLD = 0.6
CONTEXT_WINDOW = 12000
DEFAULT_CLARIFICATION_LEVEL = 5
# Pre-drill keys
PRE_DRILL_DONE      = "pre_drill_done"
AWAITING_PRE_DRILL  = "awaiting_pre_drill"
PRE_DRILL_QUERY     = "pre_drill_query"
PRE_DRILL_NODES     = "pre_drill_nodes"
DOC_CHOICES_KEY     = "doc_choices"

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
Settings.context_window = CONTEXT_WINDOW

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
            "model": "mistralai/mistral-small-3.2-24b-instruct",
            "api_base": "https://openrouter.ai/api/v1",
            "api_key": "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",
            "is_chat_model": True,
            "is_function_calling_model": False,
            "temperature": 0.2,
            "http_client": httpx.Client(verify=False),
        },
    },
}

# ─── Add this helper at the top of your file ───
async def send_animated_message(
    base_msg: str,
    frames: list,
    interval: float = 0.8
) -> None:
    """Displays an animated message optimized for performance."""
    msg = cl.Message(content=base_msg, author="Customer Service Agent")
    await msg.send()

    progress = 0
    bar_length = 12

    try:
        while True:
            current_frame = frames[progress % len(frames)]
            progress_bar = ("▣" * (progress % bar_length)).ljust(bar_length, "▢")
            # Update the content property, then issue a plain update()
            msg.content = f"{current_frame} {base_msg}\n{progress_bar}"
            await msg.update()
            progress += 1
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        # Final static display when the task is cancelled
        msg.content = base_msg
        await msg.update()
        
# ✅ Add this for on-demand manual retrieval testing
def manual_retrieve(query: str, top_k=5):
    from llama_index import Settings, VectorStoreIndex
    from llama_index.embeddings.cohere import CohereEmbedding

    Settings.embed_model = CohereEmbedding(
        api_key=os.getenv("COHERE_API_KEY"),
        model_name=os.getenv("COHERE_MODEL_ID"),
        input_type="search_document",
        embedding_type="float"
    )

    dataset = DATASET_MAPPING.get("Accounting Compliance")  # or any profile
    vector_store = qdrant_manager.get_vector_store(dataset, hybrid=True)
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    # หลังเรียก nodes = retriever.retrieve(query_with_context)
    for i, n in enumerate(nodes[:3], 1):
        logger.info(f"🏷 Top #{i}: source={n.node.metadata.get('source')} score={n.score:.3f}")

    for i, n in enumerate(nodes):
        print(f"\n== Chunk {i+1} ==")
        print("📄 Source:", n.node.metadata.get("source"))
        print(n.node.text[:800], "...\n")

    return nodes

# Load predefined answers
with open("predefined_answers.json", "r", encoding="utf-8") as f:
    predefined_answers = json.load(f)

# Global state trackers
shown_admin_replies = {}
shown_admin_reply_ids = {}
shown_parent_keys = set()
# Pre-drill flags
PRE_DRILL_KEY       = "pre_drill_done"
AWAITING_PRE_DRILL  = "awaiting_pre_drill"
DOC_CHOICES_KEY     = "doc_choices"

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


async def send_with_feedback(
    content: str,
    author: str = "Customer Service Agent",
    parent_id: str = None,
    metadata: Optional[Dict] = None,
):
    """Sends a message and streams it character by character, always reminding the user they can restart."""
    # Build footer dynamically
    footer_lines = []
    current_doc = cl.user_session.get("current_doc")
    if current_doc:
        footer_lines.append(f"📄 กำลังเช็คจากงานเอกสาร: {current_doc}")
    footer_lines.append("หากต้องการเริ่มคำถามใหม่ กรุณาพิมพ์ 0 ")
    footer = "\n\n" + "\n".join(footer_lines)

    content = content + footer

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

    if chat_profile == "Accounting Compliance 2":
        return OpenAILike(**settings)
    elif chat_profile == "Accounting Compliance":
        return OpenAILike(**settings)
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

    # Build the index using your document‐style embeddings
    index = VectorStoreIndex.from_vector_store(vector_store)
    llm = get_llm_settings(chat_profile)

    # Create the query engine, unchanged
    query_engine = index.as_query_engine(
        retriever_mode="hybrid",
        llm=llm,
        streaming=True,
        verbose=True,
        similarity_top_k=12,
        sparse_top_k=20,
        alpha=0.2,
    )

    # Create a retriever that uses the document embeddings for the index
    # but a dedicated "search_query" embedding model for query vectors
    retriever = index.as_retriever(
        similarity_top_k=12,
        # Override only the query‐side embedding model:
        embedding_model=CohereEmbedding(
            api_key=os.getenv("COHERE_API_KEY"),
            model_name=os.getenv("COHERE_MODEL_ID"),
            input_type="search_document",      # ← short‐query embedding
            embedding_type="float",
        ),
    )

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
    for key in [
        "awaiting_clarification",
        "clarification_rounds",
        "fuzzy_clarification_rounds",
        "possible_summaries",
        "nodes_to_consider",
        "summary_to_meta",
        "original_query",
        "clarification_level",   # ← depth marker
        "auto_skipped",          # ← your “skipped once” flag
        "clarification_just_exited",
        "last_was_clarify",
        "filtered_nodes",
        "hier_sections",
    ]:
        cl.user_session.set(key, None)


async def answer_from_node(node, user_q):
    clear_clarification_state()
    cl.user_session.set("awaiting_clarification", False)
    """Builds and sends the final LLM response from a single selected node with a loading animation."""
    # Always pull the original question from session if available
    orig_q = cl.user_session.get("pre_drill_query", user_q)

    src = node.node.metadata.get("source", "Unknown")
    txt = node.node.text.replace("\n", " ")
    prompt = (
        f'ผู้ใช้ถามว่า: "{orig_q}"\n'
        f'บทความที่เลือกมาจากเอกสารนโยบาย "{src}" (เต็มข้อความ):\n"""{txt}\n"""\n\n'
        "กรุณาตอบโดยอาศัยเนื้อหาในบทความนี้"
    )
    runnable = cl.user_session.get("runnable")

    # ─── Kick off the loading animation ───
    frames = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
    animation_task = asyncio.create_task(
        send_animated_message("กำลังเช็ค Policy ให้อยู่ รอสักครู่นะคะ …", frames, interval=0.3)
    )
    logger.info("🔄 Starting loading animation for answer_from_node")

    try:
        # offload the blocking .query call so the event loop stays free
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, runnable.query, prompt)
        answer = resp.response if hasattr(resp, "response") else "".join(resp.response_gen)
    finally:
        # ─── Stop the animation ───
        animation_task.cancel()
        with suppress(asyncio.CancelledError):
            await animation_task

    answer = extract_and_format_table(answer.strip())

    final = f"✅ นี่คือสิ่งที่พบจาก “{src}”:\n\n{answer}"
    await send_with_feedback(final, metadata={"difficulty": "Clarified"})
    save_conversation_log(
        cl.context.session.thread_id,
        None,
        "bot",
        answer,
        difficulty="Clarified"
    )


# ======================================================================================
# Chainlit Event Handlers
# ======================================================================================

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Handles user authentication."""
    if (username, password) == ("admin", "admin"):
        logger.info("✅ Login success for admin")
        return cl.User(
            identifier="admin",
            metadata={
                "role": "ADMIN",
                "email": "chatbot_admin@gmail.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_1", "123456"):
        logger.info("✅ Login success for User_1")
        return cl.User(
            identifier="User_1",
            metadata={
                "role": "USER",
                "email": "user_1@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_2", "123456"):
        logger.info("✅ Login success for User_2")
        return cl.User(
            identifier="User_2",
            metadata={
                "role": "USER",
                "email": "user_2@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_3", "123456"):
        logger.info("✅ Login success for User_3")
        return cl.User(
            identifier="User_3",
            metadata={
                "role": "USER",
                "email": "user_3@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_4", "123456"):
        logger.info("✅ Login success for User_4")
        return cl.User(
            identifier="User_4",
            metadata={
                "role": "USER",
                "email": "user_4@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_5", "123456"):
        logger.info("✅ Login success for User_5")
        return cl.User(
            identifier="User_5",
            metadata={
                "role": "USER",
                "email": "user_5@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_6", "123456"):
        logger.info("✅ Login success for User_6")
        return cl.User(
            identifier="User_6",
            metadata={
                "role": "USER",
                "email": "user_6@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_7", "123456"):
        logger.info("✅ Login success for User_7")
        return cl.User(
            identifier="User_7",
            metadata={
                "role": "USER",
                "email": "user_7@example.com",
                "provider": "credentials"
            }
        )

    logger.warning(f"❌ Login failed for {username}")
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
        cl.Starter(label="อำนาจอนุมัติการลงทุน investment project แต่ละประเภท แต่ละมูลค่า", message="อำนาจอนุมัติการลงทุน investment project แต่ละประเภท แต่ละมูลค่า", icon="/public/star.svg"),
        cl.Starter(label="เอกสารที่ต้องใช้สำหรับการเปิด new vendor code มีอะไรบ้าง ?", message="เอกสารที่ต้องใช้สำหรับการเปิด new vendor code มีอะไรบ้าง ?", icon="/public/star.svg"),
        cl.Starter(label="รอบการทำเบิกเงินทดรองจ่าย และการจ่ายเงิน", message="รอบการทำเบิกเงินทดรองจ่าย และการจ่ายเงิน", icon="/public/star.svg"),
        cl.Starter(label="เมื่อไรต้องเปิด PR ผ่านระบบ เมื่อไรสามารถใช้ PO manual (PO กระดาษได้)", message="เมื่อไรต้องเปิด PR ผ่านระบบ เมื่อไรสามารถใช้ PO manual (PO กระดาษได้)", icon="/public/star.svg"),
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
    text = message.content.strip()
    

    # ─── Global “start new conversation” shortcut ───
    if text == "0" or text == "❌ ถามคำถามใหม่":
        # Clear any in‐flight clarification state
        cl.user_session.set("awaiting_clarification", False)
        cl.user_session.set("clarification_just_exited", True)
        cl.user_session.set("filtered_nodes", None)
        cl.user_session.set("clarification_level", None)
        cl.user_session.set("auto_skipped", None)
        cl.user_session.set("current_doc", None)

        # Wipe Redis‐backed chat store
        thread_id = cl.context.session.thread_id
        user_id = cl.user_session.get("user").identifier
        redis_key = f"{user_id}:{thread_id}"
        redis_client.delete(redis_key)

        # Re‐init memory buffer
        fresh_mem = ChatMemoryBuffer.from_defaults(
            token_limit=TOKEN_LIMIT,
            chat_store=chat_store,
            chat_store_key=redis_key
        )
        cl.user_session.set("memory", fresh_mem)

        # Notify user
        await send_with_feedback(
            "✅ เริ่มต้นการสนทนาใหม่—หน่วยความจำถูกล้างแล้ว กรุณาพิมพ์คำถามของคุณอีกครั้ง"
        )
        return

    logger.info(f"Received message from user: {message.content}")
    runnable = cl.user_session.get("runnable")
    retriever = cl.user_session.get("retriever")
    thread_id = cl.context.session.thread_id

    save_conversation_log(thread_id, message.id, role="user", content=message.content)
    memory = cl.user_session.get("memory")
    memory.put(ChatMessage(role="user", content=message.content))

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
    """Handles user's response during a clarification flow, including hierarchical clarification."""
    # ─── Hierarchical pick response ───
    if cl.user_session.get("awaiting_clarification"):
        sections: Dict[str, List] = cl.user_session.get("hier_sections", {})
        titles = list(sections.keys())
        choice = message.content.strip()

        # Build candidates: section titles + exit option
        exit_label = "❌ ถามคำถามใหม่"
        candidates = titles + [exit_label]
        idx = None

        # Parse numeric or fuzzy choice
        if choice.isdigit():
            idx = int(choice) - 1
        else:
            from difflib import SequenceMatcher
            ratios = [SequenceMatcher(None, choice, c).ratio() for c in candidates]
            if ratios:
                max_ratio = max(ratios)
                if max_ratio > 0.6:
                    idx = ratios.index(max_ratio)

        # Validate choice
        if idx is None or idx < 0 or idx >= len(candidates):
            await send_with_feedback("⚠️ โปรดระบุหมายเลขหรือชื่อหัวข้อให้ถูกต้องอีกครั้ง")
            return

        # Exit option selected?
        if idx == len(candidates) - 1:
            # ─── User chose to restart ───
            cl.user_session.set("awaiting_clarification", False)
            cl.user_session.set("clarification_just_exited", True)
            cl.user_session.set("filtered_nodes", None)
            cl.user_session.set("clarification_level", None)

            # ─── Clear Redis chat store and reset memory buffer ───
            thread_id = cl.context.session.thread_id
            user_id = cl.user_session.get("user").identifier
            redis_key = f"{user_id}:{thread_id}"
            redis_client.delete(redis_key)
            new_mem = ChatMemoryBuffer.from_defaults(
                token_limit=TOKEN_LIMIT,
                chat_store=chat_store,
                chat_store_key=redis_key
            )
            cl.user_session.set("memory", new_mem)

            # ─── Let user know we’ve restarted ───
            await send_with_feedback(
                "✅ คุณได้เลือกเริ่มต้นคำถามใหม่แล้ว! กรุณาพิมพ์คำถามใหม่ของคุณได้เลย"
            )
            return

        # Normal section picked
        selected_title = titles[idx]
        selected_nodes = sections[selected_title]
        logger.info(f"🔍 Hierarchical: user picked “{selected_title}” with {len(selected_nodes)} chunks")
        cl.user_session.set("filtered_nodes", selected_nodes)
        # Stay in clarification flow
        cl.user_session.set("awaiting_clarification", True)

        # Bump to next deeper level
        clar_level = cl.user_session.get("clarification_level") or DEFAULT_CLARIFICATION_LEVEL
        cl.user_session.set("clarification_level", clar_level + 1)

        # Re-run standard query on filtered nodes
        return await handle_standard_query(message)
    # ─── End hierarchical pick response ───

    # 🛡️ Short-circuit: prevent re-entering after user just exited clarification
    if cl.user_session.get("clarification_just_exited"):
        logger.warning("⛔ clarification_just_exited is True — skipping clarification logic")
        return

    nodes_to_consider = cl.user_session.get("nodes_to_consider", [])
    summaries = cl.user_session.get("possible_summaries", [])
    original_query = cl.user_session.get("original_query", "")
    rounds = cl.user_session.get("clarification_rounds", 0)

    if rounds >= MAX_CLARIFICATION_ROUNDS:
        summary_to_meta = cl.user_session.get("summary_to_meta", {})
        fuzzy_candidates = [
            (q, s[2]) for q, s in summary_to_meta.items()
            if isinstance(s, tuple) and s[0] == "fuzzy"
        ]

        if fuzzy_candidates:
            best_question, score = max(fuzzy_candidates, key=lambda x: x[1])
            answer = predefined_answers.get(best_question, "")
            await send_with_feedback(
                f"{answer}\n\n🧠 DEBUG | Easy (Auto-picked fuzzy after max rounds) | Score: {score:.2f}"
            )
            save_conversation_log(cl.context.session.thread_id, None, "bot", answer, "Easy")
            clear_clarification_state()
            return

        if nodes_to_consider:
            chosen = max(nodes_to_consider, key=lambda n: n.score)
            logger.info(f"[clarify] Max rounds reached, auto-selecting node with score {chosen.score:.2f}")
            clear_clarification_state()
            return await answer_from_node(chosen, original_query)

        await send_with_feedback("⚠️ ไม่พบข้อมูลที่เกี่ยวข้อง กรุณาพิมพ์คำถามใหม่")
        clear_clarification_state()
        return

    # Increment clarification round
    cl.user_session.set("clarification_rounds", rounds + 1)
    choice = message.content.strip()
    opt_out_label = "❌ ถามคำถามใหม่"

    # Handle opt‐out outside hierarchical flow
    if (choice.isdigit() and opt_out_label in summaries and
        int(choice) - 1 == summaries.index(opt_out_label)
    ) or choice.strip().lower() in [opt_out_label, "❌", "exit", "ถามคำถามใหม่"]:
        clear_clarification_state()
        cl.user_session.set("clarification_just_exited", True)
        await send_with_feedback("✅ คุณได้เลือกเริ่มต้นคำถามใหม่ กรุณาถามคำถามของคุณอีกครั้ง")
        return

    if choice.lower() == "auto":
        chosen = max(nodes_to_consider, key=lambda n: n.score)
        logger.info(f"[clarify] User requested auto, selecting node with score {chosen.score:.2f}")
        clear_clarification_state()
        return await answer_from_node(chosen, original_query)

    # Determine selected index for summaries
    selected_index = None
    if choice.isdigit() and 0 <= int(choice) - 1 < len(summaries):
        selected_index = int(choice) - 1
    else:
        ratios = [SequenceMatcher(None, choice.lower(), s.lower()).ratio() for s in summaries]
        if ratios and max(ratios) > 0.6:
            selected_index = ratios.index(max(ratios))

    if selected_index is None:
        await send_with_feedback("⚠️ ไม่พบหัวข้อที่เลือก โปรดพิมพ์หมายเลขหรือชื่อหัวข้อให้ถูกต้องอีกครั้ง")
        return

    # Tie-breaking and fuzzy auto-pick
    summary_to_meta = cl.user_session.get("summary_to_meta", {})
    fuzzy_rounds = cl.user_session.get("fuzzy_clarification_rounds", 0)

    if fuzzy_rounds >= MAX_FUZZY_CLARIFICATION_ROUNDS:
        fuzzy_candidates = [
            (q, s[2]) for q, s in summary_to_meta.items()
            if isinstance(s, tuple) and s[0] == "fuzzy"
        ]
        if fuzzy_candidates:
            best_question, score = max(fuzzy_candidates, key=lambda x: x[1])
            answer = predefined_answers.get(best_question, "")
            await send_with_feedback(
                f"{answer}\n\n🧠 DEBUG | Easy (Auto-picked fuzzy) | Score: {score:.2f}"
            )
            save_conversation_log(cl.context.session.thread_id, None, "bot", answer, "Easy")
            clear_clarification_state()
            return
        else:
            await send_with_feedback("⚠️ ไม่พบคำถามสำเร็จรูปที่ตรง กรุณาลองถามใหม่")
            clear_clarification_state()
            return

    # If user selected a predefined fuzzy summary
    if isinstance(summary_to_meta.get(summaries[selected_index]), tuple) and \
       summary_to_meta[summaries[selected_index]][0] == "fuzzy":
        _, answer, score = summary_to_meta[summaries[selected_index]]
        await send_with_feedback(
            f"{answer}\n\n🧠 *DEBUG* | Category: **Easy (Clarified)** | Method: **Predefined** | Fuzzy: {score:.2f}",
            metadata={"difficulty": "Easy"},
        )
        save_conversation_log(cl.context.session.thread_id, None, "bot", answer, "Easy")
        clear_clarification_state()
        return

    # Fallback to vector node answer
    if nodes_to_consider and selected_index < len(nodes_to_consider):
        chosen_node = nodes_to_consider[selected_index]
        await answer_from_node(chosen_node, original_query)
    elif nodes_to_consider:
        chosen_node = max(nodes_to_consider, key=lambda n: n.score)
        await answer_from_node(chosen_node, original_query)
    else:
        await send_with_feedback(
            "⚠️ ไม่พบเนื้อหาที่เกี่ยวข้อง โปรดลองเลือกหัวข้อใหม่หรือลองถามใหม่อีกครั้ง"
        )
    clear_clarification_state()

async def re_clarify(nodes: list, original_query: str):
    """Asks for another round of clarification on a smaller set of nodes using a single LLM call."""
    llm = get_llm_settings(cl.user_session.get("chat_profile"))

    truncs = []
    node_map = {}

    for i, n in enumerate(nodes, 1):
        section_title = n.node.metadata.get("section_title", "")
        body_preview = n.node.text[:1000].strip().replace("\n", " ")
        preview_text = f"หัวข้อ: {section_title}\n{body_preview}" if section_title else body_preview
        truncs.append(f"({i}) {preview_text}")
        node_map[str(i)] = n


    # Add memory history
    memory: ChatMemoryBuffer = cl.user_session.get("memory")
    prior_messages = memory.get()
    logger.info("🧠 Chat Memory Content:")
    for m in prior_messages:
        logger.info(f"{m.role.upper()}: {m.content.strip()}")
    history_snippets = "\n".join(f"{m.role.title()}: {m.content.strip()}" for m in prior_messages if m.content.strip())

    batched_prompt = (
        f'ผู้ใช้ถามว่า: "{original_query}"\n\n'
        f"📜 ประวัติการสนทนา:\n{history_snippets}\n\n"
        f"ต่อไปนี้คือเนื้อหาจากหลายเอกสารที่อาจเกี่ยวข้อง:\n\n"
        + "\n\n".join(truncs)
        + "\n\nกรุณาสรุปแต่ละย่อหน้าเป็นหัวข้อย่อยไม่เกิน 10 คำ โดยใช้หมายเลขเดียวกับเนื้อหา เช่น (1) กรณี..., (2) กรณี..., เป็นต้น"
    )

    resp = llm.chat([ChatMessage(role="user", content=batched_prompt)])
    lines = resp.message.content.strip().splitlines()

    new_summaries = []
    new_meta = {}

    for line in lines:
        match = re.match(r"\(?(\d+)\)?[\.、:]?\s*(.*)", line)
        if match:
            idx, summary = match.groups()
            if idx in node_map and summary not in new_meta:
                new_summaries.append(summary)
                new_meta[summary] = node_map[idx]

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
    """Handles a standard, non-clarification query with hierarchical clarification."""
    import re
    from collections import defaultdict
    from difflib import SequenceMatcher
    import statistics

    # ─── A) If the user just restarted (via ❌ or 0), clear pre-drill so next input re-prompts ───
    if cl.user_session.get("clarification_just_exited"):
        cl.user_session.set(PRE_DRILL_KEY, False)
        cl.user_session.set(AWAITING_PRE_DRILL, False)
        cl.user_session.set("filtered_nodes", None)
        cl.user_session.set("pre_drill_nodes", None)
        cl.user_session.set("pre_drill_query", None)
        cl.user_session.set(DOC_CHOICES_KEY, None)
        cl.user_session.set("clarification_just_exited", False)

    # ─── 1) Pre-drill: pick the document ───
    if not cl.user_session.get(PRE_DRILL_KEY) and not cl.user_session.get(AWAITING_PRE_DRILL):
        original_q = message.content.strip()
        cl.user_session.set("pre_drill_query", original_q)

        # ─── Use high-K retriever for pre-drill so we get every H3 chunk ───
        import os
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.cohere import CohereEmbedding

        dataset = DATASET_MAPPING.get(cl.user_session.get("chat_profile"))
        vector_store = qdrant_manager.get_vector_store(dataset, hybrid=True)
        index = VectorStoreIndex.from_vector_store(vector_store)
        pre_drill_retriever = index.as_retriever(
            similarity_top_k=500,
            embedding_model=CohereEmbedding(
                api_key=os.getenv("COHERE_API_KEY"),
                model_name=os.getenv("COHERE_MODEL_ID"),
                input_type="search_document",
                embedding_type="float",
            ),
        )
        all_nodes = pre_drill_retriever.retrieve(original_q)
        cl.user_session.set("pre_drill_nodes", all_nodes)
        
        # — Log each document’s best score —
        doc_scores = {}
        for n in all_nodes:
            src = n.node.metadata.get("source", "Unknown")
            doc_scores[src] = max(doc_scores.get(src, 0.0), n.score)
        for src, score in doc_scores.items():
            logger.info(f"🔍 Doc candidate: '{src}' with top score {score:.3f}")


        # ─── AUTO-SELECT Policy FAQ.docx if confident ───

        POLICY_AUTO_THRESH    = 0.6   # only policy FAQ ≥0.55 auto-selects
        DOC_CANDIDATE_THRESH  = 0.52   # any doc ≥0.40 is eligible for the user to choose

        policy_score = doc_scores.get("Policy FAQ.docx", 0.0)
        top_score = max(doc_scores.values(), default=0.0)

        if policy_score == top_score and policy_score >= POLICY_AUTO_THRESH:
            # High-confidence hit in Policy FAQ.docx → pick it immediately
            logger.info(f"✅ Auto-selected 'Policy FAQ.docx' (score {policy_score:.3f})")
            filtered = [
                n for n in all_nodes
                if n.node.metadata.get("source") == "Policy FAQ.docx"
            ]
            cl.user_session.set("filtered_nodes", filtered)
            cl.user_session.set(PRE_DRILL_KEY, True)
            # mark that we auto-selected Policy FAQ so we can bypass auto-drill
            cl.user_session.set("policy_auto_select", True)

        else:
            # Fallback: run original multi-doc selection logic
            doc_scores = defaultdict(float)
            for n in all_nodes:
                src = n.node.metadata.get("source", "Unknown")
                doc_scores[src] = max(doc_scores[src], n.score)

            # Only docs ≥ threshold are candidates
            # Only docs ≥ DOC_CANDIDATE_THRESH *and* not the FAQ are candidates
            doc_set = [
                src for src, score in doc_scores.items()
                if score >= DOC_CANDIDATE_THRESH and src != "Policy FAQ.docx"
            ]
            for src in doc_set:
                logger.info(f"✅ Candidate doc: '{src}' (score {doc_scores[src]:.3f})")

            if len(doc_set) > 1:
                cl.user_session.set(AWAITING_PRE_DRILL, True)
                cl.user_session.set(DOC_CHOICES_KEY, doc_set)
                options = "\n".join(f"{i+1}. {d}" for i, d in enumerate(doc_set))
                await send_with_feedback(
                    f"❓ คำถามของคุณเกี่ยวกับเนื้อหาหลายเอกสาร โปรดเลือกเอกสารที่ตรงกับความต้องการ:\n\n{options}\n\n"
                    "ตอบด้วยหมายเลข เช่น `1` หรือชื่อเอกสาร",
                    author="Customer Service Agent"
                )
                return

            # Single candidate → auto-pick it
            cl.user_session.set(PRE_DRILL_KEY, True)
            if doc_set:
                single = doc_set[0]
                filtered = [
                    n for n in all_nodes
                    if n.node.metadata.get("source") == single
                ]
                cl.user_session.set("filtered_nodes", filtered)
        # ────────────────────────────────────────────────────────────────────


 

    # ─── 2) Handle the user’s document choice ───
    if cl.user_session.get(AWAITING_PRE_DRILL):
        choice = message.content.strip()
        docs = cl.user_session.get(DOC_CHOICES_KEY) or []
        idx = None

        if choice.isdigit():
            i = int(choice) - 1
            if 0 <= i < len(docs):
                idx = i
        else:
            ratios = [
                SequenceMatcher(None, choice.lower(), d.lower()).ratio()
                for d in docs
            ]
            if ratios and max(ratios) > 0.6:
                idx = ratios.index(max(ratios))

        if idx is None:
            await send_with_feedback("⚠️ โปรดระบุหมายเลขหรือชื่อเอกสารให้ถูกต้องอีกครั้ง")
            return

        selected_doc = docs[idx]
        cl.user_session.set("current_doc", selected_doc)
        cl.user_session.set(PRE_DRILL_KEY, True)
        cl.user_session.set(AWAITING_PRE_DRILL, False)

        # ─── Re-retrieve using high-K retriever so we get every H3 chunk ───
        import os
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.cohere import CohereEmbedding

        dataset = DATASET_MAPPING.get(cl.user_session.get("chat_profile"))
        vector_store = qdrant_manager.get_vector_store(dataset, hybrid=True)
        index = VectorStoreIndex.from_vector_store(vector_store)
        doc_retriever = index.as_retriever(
            similarity_top_k=500,  # fetch up to 500 chunks
            embedding_model=CohereEmbedding(
                api_key=os.getenv("COHERE_API_KEY"),
                model_name=os.getenv("COHERE_MODEL_ID"),
                input_type="search_document",
                embedding_type="float",
            ),
        )

        # Use the original question so we get all sections of that doc
        query = cl.user_session.get("pre_drill_query") or message.content
        retrieved_nodes = doc_retriever.retrieve(query)

        # Now filter down to just the user-selected document
        filtered = [
            n for n in retrieved_nodes
            if n.node.metadata.get("source") == selected_doc
        ]

        logger.info("📄 User selected doc: %s", selected_doc)
        logger.info("🔍 Retrieved %d chunks for that doc", len(filtered))

        cl.user_session.set("pre_drill_nodes", filtered)
        cl.user_session.set("filtered_nodes", filtered)
        # message.content remains unchanged so H2/H3 logic fires normally


    # ─── 3) Reset on new question ───
    if not cl.user_session.get("awaiting_clarification"):
        cl.user_session.set("auto_skipped", False)
        cl.user_session.set("hier_sections", None)
        cl.user_session.set("clarification_level", None)
        if not cl.user_session.get(PRE_DRILL_KEY):
            cl.user_session.set("filtered_nodes", None)

    # ─── 4) Prepare retrieval █────────────────────────────────────
    retriever = cl.user_session.get("retriever")
    thread_id = cl.context.session.thread_id
    memory = cl.user_session.get("memory")
    past = memory.get()[-3:]
    context = "\n".join(f"{m.role.title()}: {m.content.strip()}" for m in past if m.content.strip())
    query_with_context = f"{context}\nUser: {message.content}" if context else message.content

    # ─── 5) Retrieve (or reuse filtered_nodes) █────────────────────────────────────
    nodes = cl.user_session.get("filtered_nodes")
    if nodes is None:
        try:
            nodes = retriever.retrieve(query_with_context)
            for i, n in enumerate(nodes[:3], 1):
                logger.info(f"🏷 Top #{i}: source={n.node.metadata.get('source')} score={n.score:.3f}")

            # fuzzy-name fallback...
            name_pattern = r"^[A-Za-zก-๙]+(?:\s+[A-Za-zก-๙]+)+$"
            if re.fullmatch(name_pattern, message.content.strip()):
                best_fuzzy, best_node = 0, None
                for n in nodes:
                    score = SequenceMatcher(None, message.content.strip(), n.node.text.strip()).ratio()
                    if score > best_fuzzy:
                        best_fuzzy, best_node = score, n
                if best_fuzzy >= 0.6:
                    clear_clarification_state()
                    cl.user_session.set("awaiting_clarification", False)
                    return await answer_from_node(best_node, query_with_context)
        except Exception:
            logger.exception("❌ Retrieval failed")
            await send_with_feedback("⚠️ เกิดข้อผิดพลาดในการค้นหา กรุณาลองใหม่อีกครั้ง")
            return

    # ─── 6) Scores, early-reject, auto-drill & auto-answer ─────────────────────
    top_score = nodes[0].score if nodes else 0.0
    logger.info(
        f"🧪 DEBUG | top_score={top_score:.4f}, "
        f"VECTOR_MIN={VECTOR_MIN_THRESHOLD:.4f}, "
        f"VECTOR_MEDIUM={VECTOR_MEDIUM_THRESHOLD:.4f}"
    )
    logger.warning(
        f"🔍 About to check early-reject: top_score={top_score:.4f} vs VECTOR_MIN_THRESHOLD={VECTOR_MIN_THRESHOLD:.4f}"
    )

    best_node = max(nodes, key=lambda n: n.score)
    best_path = best_node.node.metadata.get("section_path", [])
    depth = len(best_path)

    # 6a) Early-reject at top levels (depth<3)
    if (
        top_score < VECTOR_MIN_THRESHOLD
        and not cl.user_session.get("awaiting_clarification")
        and depth < 3
    ):
        await send_with_feedback(
            "❌ คำถามไม่เกี่ยวข้อง กรุณาถามใหม่",
            metadata={"difficulty": "Rejected"}
        )
        save_conversation_log(thread_id, message.id, "bot", "Rejected", "Rejected")
        return

    # 6b) Auto-drill into H3 of the highest-scoring H2 (skip H2 menu)
    VECTOR_AUTO_LEVEL3_THRESHOLD = 0.51
    AUTO_LEVEL3_DEPTH = 3
    if not cl.user_session.get("awaiting_clarification") and top_score >= VECTOR_AUTO_LEVEL3_THRESHOLD:
        # 1) Group by H2
        from collections import defaultdict
        raw_h2 = defaultdict(list)
        for n in nodes:
            path = n.node.metadata.get("section_path", [])
            if len(path) >= 2:
                raw_h2[path[1]].append(n)

        if raw_h2:
            # 2) Score each H2 by its top chunk
            section_scores = {h2: max(n.score for n in grp) for h2, grp in raw_h2.items()}
            # 3) Pick the H2 with highest score
            top_h2, top_h2_score = max(section_scores.items(), key=lambda x: x[1])
            logger.info(f"🔍 Best H2 candidate: '{top_h2}' with score {top_h2_score:.3f}")
            
            for h2, score in section_scores.items():
                logger.info(f"🔍 H2 candidate: '{h2}' → score {score:.3f}")

            # Only drill if that H2 is confident enough
            if top_h2_score >= VECTOR_AUTO_LEVEL3_THRESHOLD:
                # 4) Gather all H3 under that H2
                h3_chunks = [
                    n for n in nodes
                    if len(n.node.metadata.get("section_path", [])) >= 3
                    and n.node.metadata["section_path"][1] == top_h2
                ]
                if h3_chunks:
                    logger.info(
                        "🏷 Auto-drill triggered: skipping H2 '%s' (score %.3f) → %d H3 chunks",
                        top_h2, top_h2_score, len(h3_chunks)
                    )
                    cl.user_session.set("filtered_nodes", h3_chunks)
                    cl.user_session.set("clarification_level", AUTO_LEVEL3_DEPTH)
                    cl.user_session.set("awaiting_clarification", True)
                    return await handle_standard_query(message)

    # 6c) Auto-answer if extremely confident
    VECTOR_AUTO_DIRECT_THRESHOLD = 0.63
    if top_score >= VECTOR_AUTO_DIRECT_THRESHOLD:
        logger.info(
            "✅ Auto-answer triggered at depth %d (score %.3f)",
            depth, top_score
        )
        clear_clarification_state()
        cl.user_session.set("awaiting_clarification", False)
        return await answer_from_node(best_node, query_with_context)

    # ─── 8) 0th-drill: hierarchical section drill ─────────────────────────────────
    raw_h2 = defaultdict(list)
    for n in nodes:
        path = n.node.metadata.get("section_path", [])
        if len(path) >= 2:
            raw_h2[path[1]].append(n)
    section_scores = {h: max(c.score for c in chs) for h, chs in raw_h2.items()}
    # ─── Log each H2 candidate’s score ───
    for h2, score in section_scores.items():
        logger.info(f"🔍 H2 candidate: '{h2}' with vector score {score:.3f}")

    if cl.user_session.get("awaiting_clarification"):
        ordered_h2 = sorted(section_scores, key=lambda h: section_scores[h], reverse=True)
    else:
        ordered_h2 = sorted([h for h, s in section_scores.items() if s >= VECTOR_MIN_THRESHOLD],
                            key=lambda h: section_scores[h], reverse=True)
    logger.info(f"🔍 ordered_h2 (post-threshold): {ordered_h2}")

    # ─── 9) Show H2 menu or dive into H3 / answer as before ────────────────────────
    H2_THRESHOLD = 0.35
    MAX_H2_OPTIONS = 5

    # Filter H2s with score >= threshold
    filtered_h2s = [h for h in section_scores if section_scores[h] >= H2_THRESHOLD]

    # Sort and take top N
    ordered_h2 = sorted(filtered_h2s, key=lambda h: section_scores[h], reverse=True)[:MAX_H2_OPTIONS]

    logger.info("🔍 ordered_h2 (filtered ≥ %.2f, top %d): %s",
                H2_THRESHOLD,
                MAX_H2_OPTIONS,
                ", ".join(f"'{h}' ({section_scores[h]:.3f})" for h in ordered_h2))

    if len(ordered_h2) > 1:
        logger.info(f"🏷 Showing H2 menu with {len(ordered_h2)} options")
        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("hier_sections", {h: raw_h2[h] for h in ordered_h2})
        exit_label = "❌ ถามคำถามใหม่"
        opts = ordered_h2 + [exit_label]
        lines = [f"{i+1}. {h}" for i, h in enumerate(opts)]
        await send_with_feedback(
            f"❓ โปรดเลือกหัวข้อย่อย (ระดับ 2):\n\n" + "\n".join(lines),
            author="Customer Service Agent"
        )
        return


    # ─── 10) Exactly one H2 → drill into H3 (or answer if no children) ──────────
    if len(ordered_h2) == 1:
        h2_key = ordered_h2[0]
        logger.info("🏷 Single H2 chosen: %s", h2_key)

        raw_h3 = defaultdict(list)
        for n in nodes:
            path = n.node.metadata.get("section_path", [])
            if len(path) >= 3 and path[1] == h2_key:
                raw_h3[path[2]].append(n)
        logger.info("🔍 raw_h3 groups: %s", {h: len(chunks) for h, chunks in raw_h3.items()})

        ordered_h3 = sorted(raw_h3.keys(), key=lambda h: max(c.score for c in raw_h3[h]), reverse=True)
        logger.info("🔍 ordered_h3: %s", ordered_h3)

        if not ordered_h3:
            best_chunk = max(raw_h2[h2_key], key=lambda n: n.score)
            logger.info("✅ No H3 children — answering from H2 chunk: %s", best_chunk.node.metadata.get("section_path"))
            clear_clarification_state()
            cl.user_session.set("awaiting_clarification", False)
            return await answer_from_node(best_chunk, query_with_context)

        if len(ordered_h3) > 1:
            logger.info("🏷 Showing H3 menu with %d options", len(ordered_h3))
            cl.user_session.set("awaiting_clarification", True)
            cl.user_session.set("hier_sections", {h: raw_h3[h] for h in ordered_h3})
            cl.user_session.set("clarification_level", 3)
            exit_label = "❌ ถามคำถามใหม่"
            opts = ordered_h3 + [exit_label]
            lines = [f"{i+1}. {h}" for i, h in enumerate(opts)]
            await send_with_feedback(
                f"❓ โปรดเลือกหัวข้อย่อย (ระดับ 3):\n\n" + "\n".join(lines),
                author="Customer Service Agent"
            )
            return

        # Exactly one H3 child → answer immediately
        best_chunk = max(raw_h3[ordered_h3[0]], key=lambda n: n.score)
        logger.info("✅ Single H3 child — answering from chunk: %s", best_chunk.node.metadata.get("section_path"))
        clear_clarification_state()
        cl.user_session.set("awaiting_clarification", False)
        return await answer_from_node(best_chunk, query_with_context)

    # ─── 11) Fuzzy-fallback & final LLM answer ─────────────────────────────────
    fuzzy_scores = {q: SequenceMatcher(None, message.content.lower(), q.lower()).ratio() for q in predefined_answers}
    fuzzy_cands = sorted([(q, s) for q, s in fuzzy_scores.items() if s > FUZZY_THRESHOLD], key=lambda x: x[1], reverse=True)
    best_q, fuzzy_score = fuzzy_cands[0] if fuzzy_cands else ("", 0)
    if fuzzy_score > top_score:
        if len(fuzzy_cands) > 1:
            return await start_clarification_flow([], message.content, fuzzy_cands)
        answer = predefined_answers[best_q]
        lvl = "Easy"
        await send_with_feedback(
            f"{answer}\n\n🧠 *DEBUG* | Category: **Easy** | Method: **Predefined** | Fuzzy: {fuzzy_score:.2f} | Vector: {top_score:.2f}",
            metadata={"difficulty": lvl}
        )
        save_conversation_log(thread_id, message.id, "bot", answer, lvl)
        return

    # ─── 12) LLM answer ─────────────────────────────────────────────
    ctx2 = "\n".join(m.content for m in memory.get()[-3:] if m.role == "user")
    final_q = f"{ctx2}\n{message.content}" if ctx2 else message.content
    lvl = "Hard" if top_score >= VECTOR_MEDIUM_THRESHOLD else "Medium"
    await answer_with_llm(nodes, final_q, lvl, top_score, fuzzy_score)

    # ─── 13) Reset for next new question ─────────────────────────────────
    clear_clarification_state()
    cl.user_session.set("awaiting_clarification", False)
    cl.user_session.set(PRE_DRILL_KEY, False)

async def start_clarification_flow(nodes: list, original_query: str, fuzzy_candidates: list = None):
    """Initiates the clarification process when a query is too broad."""
    # Ensure fuzzy_clarification_rounds is initialized
    cl.user_session.set("clarification_just_exited", False)
    # If we’re not mid‐clarification, clear any leftover hierarchy state
    if not cl.user_session.get("awaiting_hier_clarification"):
        cl.user_session.set("clarification_level", None)
        cl.user_session.set("filtered_nodes", None)
    if fuzzy_candidates:
        current_round = cl.user_session.get("fuzzy_clarification_rounds") or 0
        cl.user_session.set("fuzzy_clarification_rounds", current_round + 1)
        summaries = []
        summary_to_meta = {}
        for q, score in fuzzy_candidates[:MAX_FUZZY_CLARIFY_TOPICS]:
            summaries.append(q)
            summary_to_meta[q] = ("fuzzy", predefined_answers[q], score)
            logger.info(f"🔍 Fuzzy clarification choice: {q} | Score: {score:.2f}")

        opt_out_choice = "❌ ถามคำถามใหม่"
        if opt_out_choice not in summaries:
            summaries.append(opt_out_choice)

        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("clarification_rounds", 0)
        cl.user_session.set("possible_summaries", summaries)
        cl.user_session.set("nodes_to_consider", [])  # Empty list for fuzzy
        cl.user_session.set("summary_to_meta", summary_to_meta)
        cl.user_session.set("original_query", original_query)

        await send_with_feedback(
            content=(
                "❓ คำถามของคุณอาจตรงกับหัวข้อเหล่านี้\n\n"
                + "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
                + '\n\nโปรดตอบกลับด้วยหมายเลขหรือชื่อหัวข้อที่ต้องการ หรือเลือก \"❌ ถามคำถามใหม่\" หากต้องการเริ่มต้นใหม่'
            ),
            author="Customer Service Agent",
        )
        return

    llm = get_llm_settings(cl.user_session.get("chat_profile"))
    summaries, summary_to_meta = [], {}

    nodes_to_summarize = [n for n in nodes[:MAX_TOPICS_BEFORE_CLARIFY] if any(tok in n.node.text for tok in re.findall(r"\w+", original_query))]
    if len(nodes_to_summarize) < 2:
        nodes_to_summarize = nodes[:MAX_TOPICS_BEFORE_CLARIFY]

    for n in nodes_to_summarize:

            # PREPARE BATCHED PROMPT
        truncs = []
        node_map = {}
        for i, n in enumerate(nodes_to_summarize, 1):
            trunc_text = n.node.text[:1000].strip().replace("\n", " ")
            section_title = n.node.metadata.get("section_title", "")
            if section_title:
                trunc_text = f"{section_title}\n{trunc_text}"
            truncs.append(f"({i})\n{trunc_text}")
            node_map[str(i)] = n


        # Add memory history
        memory: ChatMemoryBuffer = cl.user_session.get("memory")
        prior_messages = memory.get()
        history_snippets = "\n".join(f"{m.role.title()}: {m.content.strip()}" for m in prior_messages if m.content.strip())

        batched_prompt = (
            f'ผู้ใช้ถามว่า: "{original_query}"\n\n'
            f"📜 ประวัติการสนทนา:\n{history_snippets}\n\n"
            f"ต่อไปนี้คือเนื้อหาจากหลายเอกสารที่อาจเกี่ยวข้อง:\n\n"
            + "\n\n".join(truncs)
            + "\n\nกรุณาสรุปแต่ละย่อหน้าเป็นหัวข้อย่อยไม่เกิน 10 คำ โดยใช้หมายเลขเดียวกับเนื้อหา เช่น (1) กรณี..., (2) กรณี..., เป็นต้น"
        )

        # CALL LLM ONCE
        resp = llm.chat([ChatMessage(role="user", content=batched_prompt)])
        lines = resp.message.content.strip().splitlines()

        # MAP RESPONSES BACK TO NODES
        summaries = []
        summary_to_meta = {}
        for line in lines:
            match = re.match(r"\(?(\d+)\)?[\.、:]?\s*(.*)", line)
            if match:
                idx, summary = match.groups()
                if idx in node_map and summary not in summary_to_meta:
                    summaries.append(summary)
                    summary_to_meta[summary] = (node_map[idx], node_map[idx].node.text[:1000], node_map[idx].node.metadata.get("source", "UnknownPolicy"))
        # 🧠 Append fuzzy match questions into the clarification loop
        for i, (question, score) in enumerate(fuzzy_candidates, 1):
            label = f'✅ คำถามสำเร็จรูป: "{question}"'
            if label not in summaries:
                summaries.append(label)
                summary_to_meta[label] = ("fuzzy", predefined_answers[question], score)
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

        # 🧠 Log clarification details to terminal
        logger.info("📌 Clarification Triggered")
        logger.info(f"🔍 User Question: {original_query}")
        logger.info("📑 Selected Chunks for Clarification:")
        for i, n in enumerate(nodes_to_summarize):
            preview = n.node.text[:120].replace("\n", " ")
            logger.info(f"  {i+1}. Title: {n.node.metadata.get('section_title', 'Unknown')} | Score: {n.score:.4f} | Preview: {preview}")

        logger.info("🧠 Clarification Choices:")
        for i, s in enumerate(summaries):
            logger.info(f"  {i+1}. {s}")
            if isinstance(summary_to_meta.get(s), tuple) and summary_to_meta[s][0] == "fuzzy":
                logger.info(f"     ↳ Predefined answer score: {summary_to_meta[s][2]:.2f}")

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
    clear_clarification_state()
    cl.user_session.set("awaiting_clarification", False)
    """Generates an answer using the LLM with context from retrieved nodes."""
    runnable = cl.user_session.get("runnable")
    TOP_K = 3
    selected_nodes = nodes
    logger.info("📤 answer_with_llm contexts (final): %s", [n.node.text for n in selected_nodes])
    contexts = [
        (n.node.metadata.get("source", "Unknown"), n.node.text.strip().replace("\n", " "))
        for n in selected_nodes
    ]

    # Build chunk context
    chunk_context = "".join(
        f'({i}) เอกสารนโยบาย: "{src}"\n'
        f'เนื้อหาชิ้นนี้ (เต็มข้อความ):\n"""{txt}\n"""\n\n'
        for i, (src, txt) in enumerate(contexts, 1)
    )

    # Include prior messages from memory
    memory = cl.user_session.get("memory")
    prior_messages = memory.get()[-5:]
    history_snippets = ""
    for m in prior_messages:
        if m.role == "user":
            history_snippets += f"👤 ผู้ใช้: {m.content.strip()}\n"
        elif m.role == "assistant":
            history_snippets += f"🤖 บอท: {m.content.strip()}\n"

    logger.info("🧠 Chat Memory Used in Prompt:")
    for m in prior_messages:
        logger.info(f"{m.role}: {m.content.strip()}")

    context_str = (
        f"📜 ประวัติการสนทนา:\n{history_snippets}\n\n"
        f"📚 ข้อความจากเอกสาร:\n{chunk_context}"
    )

    # Hint for table formatting if relevant
    suggest_table = any(
        kw in txt for _, txt in contexts
        for kw in ["20 ล้านบาท", "500,000 บาท", "ประเภทที่", "โครงการ"]
    )
    formatting_hint = (
        "\n🧮 คำแนะนำสำคัญ: คำถามของผู้ใช้มีการระบุ 'มูลค่า' ที่ชัดเจน ...\n"
        "หากสามารถจัดให้อยู่ในรูปแบบ **ตาราง Markdown** ได้ ...\n"
    ) if suggest_table else "\nหากเหมาะสม ให้จัดคำตอบในรูปแบบ bullet หรือย่อหน้าเพื่อความเข้าใจง่าย"

    constraint = (
        "\n\n🔒 โปรดใช้เฉพาะข้อมูลจากชิ้นเนื้อหาด้านบนที่ส่งมาเท่านั้น "
        "และอย่าอ้างอิงเนื้อหาในส่วนอื่นๆ"
    )

    filtered_query = (
        f'ผู้ใช้ถามว่า: "{query}"\n\n'
        f"กรุณาตอบโดยอาศัยเนื้อหาต่อไปนี้ทั้งหมด:\n\n{context_str}"
        f"{formatting_hint}"
        f"{constraint}\n\n"
        "ในคำตอบให้ระบุชื่อเอกสารนโยบายที่ใช้ ... ถ้าไม่แน่ใจให้ถามกลับ"
    )

    # ─── Start the thinking animation ───
    animation_task = asyncio.create_task(
        send_animated_message(
            base_msg="กำลังเช็ค Policy ให้อยู่ รอสักครู่นะคะ...",
            frames=["🌑","🌒","🌓","🌔","🌕","🌖","🌗","🌘"],
            interval=0.3
        )
    )

    # ─── Call the LLM off the event loop ───
    try:
        resp = await asyncio.to_thread(runnable.query, filtered_query)
        answer_body = (
            resp.response.strip()
            if hasattr(resp, "response")
            else "".join(resp.response_gen).strip()
        )
    except Exception as e:
        answer_body = f"⚠️ LLM error: {e}"
    finally:
        # ─── Stop the animation ───
        animation_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await animation_task

    answer_body = extract_and_format_table(answer_body)

    # Render a clean markdown table if present
    if "|" in answer_body and "---" in answer_body:
        answer_body = f"**คำตอบ**\n\n{answer_body.strip()}"

    final_answer = (
        f"{answer_body}\n\n"
        f"🧠 *DEBUG* | Category: **{level}** | "
        f"Method: **VectorStore + LLM (top {TOP_K})** | "
        f"Vector: {top_score:.2f} | Fuzzy: {fuzzy_score:.2f}"
    )

    # Send & log
    memory.put(ChatMessage(role="assistant", content=answer_body))
    await send_with_feedback(final_answer, metadata={"difficulty": level})
    save_conversation_log(
        cl.context.session.thread_id,
        cl.context.session.id,
        "bot",
        final_answer,
        level
    )

    # ─── Auto‐clear entire chat memory after LLM reply ───
    thread_id = cl.context.session.thread_id
    user_id   = cl.user_session.get("user").identifier
    redis_key = f"{user_id}:{thread_id}"
    redis_client.delete(redis_key)
    fresh_mem = ChatMemoryBuffer.from_defaults(
        token_limit=TOKEN_LIMIT,
        chat_store=chat_store,
        chat_store_key=redis_key,
    )
    cl.user_session.set("memory", fresh_mem)

    # Reset any leftover hierarchical state
    cl.user_session.set("clarification_level", None)
    cl.user_session.set("filtered_nodes", None)
    
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