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
SELECTION_PATH_KEY = "selection_path"

logger.info(f"📡 MS_TEAMS_WORKFLOW_URL: {MS_TEAMS_WORKFLOW_URL}")
logger.info(f"✅ Loaded CHAINLIT_AUTH_SECRET: {CHAINLIT_AUTH_SECRET}")


# Constants
MAX_CLARIFICATION_ROUNDS = 2
MAX_FUZZY_CLARIFICATION_ROUNDS = 3
MAX_TOPICS_BEFORE_CLARIFY = 7
MAX_FUZZY_CLARIFY_TOPICS = 5
SIMILARITY_TIE_THRESHOLD = 0.03
FUZZY_THRESHOLD = 1.55
FUZZY_CLARIFY_THRESHOLD = 0.85  # 👈 triggers clarification when multiple fuzzy candidates exist
VECTOR_MIN_THRESHOLD = 0.3
VECTOR_MEDIUM_THRESHOLD = 0.56
CONTEXT_WINDOW = 12000
DEFAULT_CLARIFICATION_LEVEL = 5
# Pre-drill keys
PRE_DRILL_DONE      = "pre_drill_done"
AWAITING_PRE_DRILL  = "awaiting_pre_drill"
PRE_DRILL_QUERY     = "pre_drill_query"
PRE_DRILL_NODES     = "pre_drill_nodes"
DOC_CHOICES_KEY     = "doc_choices"
# Business Unit pre-drill key
SELECTED_BUSINESS_UNIT = "selected_bu"


BU_DOCUMENT_MAP = {
    "อำนาจอนุมัติ DoA / LoA การอนุมัติโครงการ และค่าใช้จ่าย": [
        "อำนาจอนุมัติ DoA และ LoA.docx",
        "เงินลงทุนในโครงการ.docx",
        "อำนาจอนุมัติรายจ่ายทั่วไป.docx",
        "การเบิกค่าใช้จ่ายพนักงาน.docx",
        "Policy FAQ.docx"
    ],
    "คู่ค้าซื้อมาขายไป (Commercial / Trade Supplier)": [
        "การเพิ่มข้อมูลคู่ค้า และการจ่ายเงิน (Trade).docx",
        "Policy FAQ.docx"
    ],
    "คู่ค้าอื่นๆ (Procurement / Non-Trade Supplier)": [
        "การเพิ่มและแก้ไขข้อมูลคู่ค้า (Non-trade).docx",
        "Policy FAQ.docx"
    ],
    "ลูกค้าผู้เช่าพื้นที่ (Mall / Tenant)": [
        "การเพิ่ม คัดเลือกลูกค้า การต่อสัญญา และการติดตามหนี้.docx",
        "Policy FAQ.docx"
    ],
    "ลูกค้า B2B": [
        "การจัดการสินเชื่อสำหรับธุรกิจ B2B.docx",
        "Policy FAQ.docx",
        "B2B Others.docx"
    ],
    "ลูกหนี้อื่นๆ (AR Others / AR non-mall)": [
        "การเพิ่มและแก้ไขข้อมูลคู่ค้า (Non-trade).docx",
        "Policy FAQ.docx"
    ],
    "สินทรัพย์ (Asset)": [
        "FA-G-13 - Asset management policy.docx",
        "Policy FAQ.docx"
    ]
}

# Redis Client

# Use TLS-enabled URL directly
redis_client = redis.Redis.from_url(
    REDIS_CHATSTORE_URI,
    decode_responses=True  # Optional: returns strings instead of bytes
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
            "model": "default",
            "api_base": "https://api-cpxis.lotuss.com/llm/v1",
            "api_key": "finance.lotuss.E9DD48B6C26A276CF48CDBC4D7468",
            "is_chat_model": True,
            "is_function_calling_model": False,
            "temperature": 0.2,
            "http_client": httpx.Client(verify=False),
        },
    },
}

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

async def ask_business_unit():
    logger.info("🟡 Triggering BU selection prompt")
    business_units = [
    "อำนาจอนุมัติ DoA / LoA การอนุมัติโครงการ และค่าใช้จ่าย",
    "คู่ค้าซื้อมาขายไป (Commercial / Trade Supplier)",
    "คู่ค้าอื่นๆ (Procurement / Non-Trade Supplier)",
    "ลูกค้าผู้เช่าพื้นที่ (Mall / Tenant)",
    "ลูกค้า B2B",
    "ลูกหนี้อื่นๆ (AR Others / AR non-mall)",
    "สินทรัพย์ (Asset)"
    ]
    options = "\n".join(f"{i+1}. {bu}" for i, bu in enumerate(business_units))
    cl.user_session.set("awaiting_bu_selection", True)
    cl.user_session.set("business_units", business_units)
    await cl.Message(content="กรุณาเลือก Business Group ที่เกี่ยวข้องกับคำถามของคุณ:\n\n" + options).send()
        
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
    nodes = retriever.retrieve(query_with_context)
    selected_bu = cl.user_session.get("selected_bu")
    allowed_docs = BU_DOCUMENT_MAP.get(selected_bu, [])
    nodes = [n for n in nodes if n.node.metadata.get("source") in allowed_docs]
    logger.info(f"📁 Filtered {len(nodes)} nodes from BU '{selected_bu}'")
    for i, n in enumerate(nodes[:3], 1):
        # grab a cleaned-up snippet of the chunk
        snippet = n.node.get_text().strip().replace("\n", " ")
        # log source, score, section path, and the snippet
        logger.info(
            "🏷 Top #%d: source=%s score=%.3f path=%s\n    chunk=\"%s\"",
            i,
            n.node.metadata.get("source"),
            n.score,
            n.node.metadata.get("section_path"),
            snippet[:200]  # first 200 chars
        )

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
    footer_lines.append("🔴 หากต้องการเริ่มคำถามใหม่ กรุณาพิมพ์ 0 ")
    footer = "\n\n" + "\n".join(footer_lines)

    content = content + footer

    msg = cl.Message(content="", author=author, parent_id=parent_id, metadata=metadata or {})
    await msg.send()
    for char in content:
        await msg.stream_token(char)
        await asyncio.sleep(0.005)
    await msg.update()

    # ✅ Save assistant message to chat memory
    memory = cl.user_session.get("memory")
    memory.put(ChatMessage(role="assistant", content=content.strip()))


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


async def answer_from_node(node_or_nodes, user_q):
    """Builds and sends the final LLM response from one or more selected nodes with a loading animation."""
    clear_clarification_state()
    cl.user_session.set("awaiting_clarification", False)

    orig_q = cl.user_session.get("original_user_question")
    runnable = cl.user_session.get("runnable")

    # Ensure input is a list of nodes
    if isinstance(node_or_nodes, list):
        nodes = node_or_nodes
    else:
        nodes = [node_or_nodes]

    # Log the number of chunks being sent
    logger.info(f"📚 Preparing to answer with {len(nodes)} chunk(s) from source.")

    # Build full text from all selected chunks
    full_text = "\n\n".join(n.node.text.strip().replace("\n", " ") for n in nodes)
    source = nodes[0].node.metadata.get("source", "Unknown")

    logger.info(f"📄 Answer source: {source}")
    logger.info(f"📦 Combined chunk text length: {len(full_text)} characters")

    # Build the LLM prompt
    section_titles = [n.node.metadata.get("section_path", [])[-1] for n in nodes]
    section_str = " / ".join(section_titles)
    # Track selection path (e.g., H1 → H2 → H3)
    full_paths = [n.node.metadata.get("section_path", []) for n in nodes]
    if full_paths:
        # Choose the most complete section path
        deepest_path = max(full_paths, key=lambda p: len(p))
        selection_path = cl.user_session.get("selection_path") or []
        selection_path.append(" / ".join(deepest_path))
        cl.user_session.set("selection_path", selection_path)
        logger.info(f"📌 Updated selection path memory: {selection_path}")

    path_history_str = "\n".join(f"👉 {p}" for p in selection_path)

    # Build the LLM prompt using selection path
    prompt = (
        f'ผู้ใช้ถามว่า: "{orig_q}"\n\n'
        f'🧭 เส้นทางการเลือกหัวข้อของผู้ใช้:\n{path_history_str}\n\n'
        f'📚 หัวข้อที่เกี่ยวข้อง: {section_str}\n'
        f'📄 เอกสารนโยบาย: "{source}"\n\n'
        f'เนื้อหาที่เกี่ยวข้องมีดังนี้:\n"""{full_text}\n"""\n\n'
        "กรุณาตอบโดยอ้างอิงรายละเอียดทั้งหมดจากเนื้อหานี้อย่างครบถ้วนและระบุเงื่อนไขที่เกี่ยวข้องให้ชัดเจน "
        "หากมีกรณีหรือเงื่อนไขพิเศษ โปรดแสดงให้ครบทุกกรณี เช่น “ถ้า…ให้…” หรือ “ในกรณีที่…ต้อง…” "
        "และอย่าสรุปรวมหลายเงื่อนไขเป็นบรรทัดเดียว"
    )
    
    logger.info(f"🧠 Final LLM prompt = \n{prompt}")

    # Loading animation
    frames = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
    animation_task = asyncio.create_task(
        send_animated_message("กำลังเช็ค Policy ให้อยู่ รอสักครู่นะคะ …", frames, interval=0.3)
    )
    logger.info("🔄 Starting loading animation for answer_from_node")

    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, runnable.query, prompt)
        answer = resp.response if hasattr(resp, "response") else "".join(resp.response_gen)
    finally:
        animation_task.cancel()
        with suppress(asyncio.CancelledError):
            await animation_task

    answer = extract_and_format_table(answer.strip())
    final = f"✅ นี่คือสิ่งที่พบจาก “{source}”:\n\n{answer}"

    # Send and log the response
    await send_with_feedback(final, metadata={"difficulty": "Clarified"})
    save_conversation_log(
        cl.context.session.thread_id,
        None,
        "bot",
        answer,
        difficulty="Clarified"
    )
    
    cl.user_session.set("reset_memory_next_turn", True)

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
        
        
    if (username, password) == ("User_8", "123456"):
        logger.info("✅ Login success for User_8")
        return cl.User(
            identifier="User_8",
            metadata={
                "role": "USER",
                "email": "user_8@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_9", "123456"):
        logger.info("✅ Login success for User_9")
        return cl.User(
            identifier="User_9",
            metadata={
                "role": "USER",
                "email": "user_9@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_10", "123456"):
        logger.info("✅ Login success for User_10")
        return cl.User(
            identifier="User_10",
            metadata={
                "role": "USER",
                "email": "user_10@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_11", "123456"):
        logger.info("✅ Login success for User_11")
        return cl.User(
            identifier="User_11",
            metadata={
                "role": "USER",
                "email": "user_11@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_12", "123456"):
        logger.info("✅ Login success for User_12")
        return cl.User(
            identifier="User_12",
            metadata={
                "role": "USER",
                "email": "user_12@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_13", "123456"):
        logger.info("✅ Login success for User_13")
        return cl.User(
            identifier="User_13",
            metadata={
                "role": "USER",
                "email": "user_13@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_14", "123456"):
        logger.info("✅ Login success for User_14")
        return cl.User(
            identifier="User_14",
            metadata={
                "role": "USER",
                "email": "user_14@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_15", "123456"):
        logger.info("✅ Login success for User_15")
        return cl.User(
            identifier="User_15",
            metadata={
                "role": "USER",
                "email": "user_15@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_16", "123456"):
        logger.info("✅ Login success for User_16")
        return cl.User(
            identifier="User_16",
            metadata={
                "role": "USER",
                "email": "user_16@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_17", "123456"):
        logger.info("✅ Login success for User_17")
        return cl.User(
            identifier="User_17",
            metadata={
                "role": "USER",
                "email": "user_17@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_18", "123456"):
        logger.info("✅ Login success for User_18")
        return cl.User(
            identifier="User_18",
            metadata={
                "role": "USER",
                "email": "user_18@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_19", "123456"):
        logger.info("✅ Login success for User_19")
        return cl.User(
            identifier="User_19",
            metadata={
                "role": "USER",
                "email": "user_19@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_20", "123456"):
        logger.info("✅ Login success for User_20")
        return cl.User(
            identifier="User_20",
            metadata={
                "role": "USER",
                "email": "user_20@example.com",
                "provider": "credentials"
            }
        )
        
    if (username, password) == ("User_21", "123456"):
        logger.info("✅ Login success for User_21")
        return cl.User(
            identifier="User_21",
            metadata={
                "role": "USER",
                "email": "user_21@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_22", "123456"):
        logger.info("✅ Login success for User_22")
        return cl.User(
            identifier="User_22",
            metadata={
                "role": "USER",
                "email": "user_22@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_23", "123456"):
        logger.info("✅ Login success for User_23")
        return cl.User(
            identifier="User_23",
            metadata={
                "role": "USER",
                "email": "user_23@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_24", "123456"):
        logger.info("✅ Login success for User_24")
        return cl.User(
            identifier="User_24",
            metadata={
                "role": "USER",
                "email": "user_24@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_25", "123456"):
        logger.info("✅ Login success for User_25")
        return cl.User(
            identifier="User_25",
            metadata={
                "role": "USER",
                "email": "user_25@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_26", "123456"):
        logger.info("✅ Login success for User_26")
        return cl.User(
            identifier="User_26",
            metadata={
                "role": "USER",
                "email": "user_26@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_27", "123456"):
        logger.info("✅ Login success for User_27")
        return cl.User(
            identifier="User_27",
            metadata={
                "role": "USER",
                "email": "user_27@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_28", "123456"):
        logger.info("✅ Login success for User_28")
        return cl.User(
            identifier="User_28",
            metadata={
                "role": "USER",
                "email": "user_28@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_29", "123456"):
        logger.info("✅ Login success for User_29")
        return cl.User(
            identifier="User_29",
            metadata={
                "role": "USER",
                "email": "user_29@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_30", "123456"):
        logger.info("✅ Login success for User_30")
        return cl.User(
            identifier="User_30",
            metadata={
                "role": "USER",
                "email": "user_30@example.com",
                "provider": "credentials"
            }
        )



    logger.warning(f"❌ Login failed for {username}")
    


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

    # 🛡️ Ensure user exists if login was skipped or logout occurred
    if not cl.user_session.get("user"):
        cl.user_session.set("user", cl.User(identifier="guest"))

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

    # ✅ Reset chat profile state
    cl.user_session.set("chat_profile", {
        "name": "default",
        "clarify_state": None,
        "clarification_candidates": [],
        "selected_clarification": None,
        "question_count": 0,
        "current_bu": None
    })

    # ✅ Clear clarification state
    clear_clarification_state()

    logger.info("🚀 on_chat_start triggered")

    await ask_business_unit()


@cl.on_message
async def on_message(message: cl.Message):
    """Handles incoming user messages."""
    text = message.content.strip()
    thread_id = cl.context.session.thread_id

    # ✅ Log incoming message
    save_conversation_log(thread_id, message.id, role="user", content=text)

    # ✅ Ensure memory exists
    memory = cl.user_session.get("memory")
    if memory is None:
        user_id = cl.user_session.get("user").identifier
        redis_key = f"{user_id}:{thread_id}"
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=TOKEN_LIMIT,
            chat_store=chat_store,
            chat_store_key=redis_key
        )
        cl.user_session.set("memory", memory)

    # ✅ Append new message to memory
    # ✅ Only log raw user input if not in clarification mode
    if not cl.user_session.get("awaiting_clarification"):
        memory.put(ChatMessage(role="user", content=text))
        logger.info(f"✅ Appended to memory: {text}")

    # ✅ Log updated memory state
    logger.info("🧠 Memory after appending new user message:")
    for i, msg in enumerate(memory.get()):
        logger.info(f"[{i}] {msg.role.upper()}: {msg.content}")
    
    
    if cl.user_session.get("selected_bu") is None and not cl.user_session.get("awaiting_bu_selection"):
        logger.info("💡 First user message with no BU selected → ask for BU")
        await ask_business_unit()
        return
    # Handle BU selection if awaiting
    if cl.user_session.get("awaiting_bu_selection", False):
        logger.info("🔁 Handling user BU input: %s", text)
        bu_list = cl.user_session.get("business_units") or []
        try:
            index = int(text) - 1
            if 0 <= index < len(bu_list):
                selected_bu = bu_list[index]
                cl.user_session.set("selected_bu", selected_bu)
                cl.user_session.set("awaiting_bu_selection", False)
                logger.info("✅ BU selected: %s", selected_bu)
                await cl.Message(content=f"✅ เลือก BU: {selected_bu} แล้ว กรุณาพิมพ์คำถามของคุณ").send()
                return  # <-- keep return only here after successful BU selection
            else:
                logger.warning("⚠️ Invalid BU index")
                await cl.Message(content="⚠️ โปรดเลือกหมายเลขที่ถูกต้อง").send()
                return
        except ValueError:
            logger.warning("⚠️ Non-numeric BU input")
            await cl.Message(content="⚠️ โปรดระบุหมายเลขของ BU ที่ต้องการ").send()
            return

    # ─── Global “start new conversation” shortcut ───
    if text == "0" or text == "❌ ถามคำถามใหม่":
        clear_clarification_state()

        for key in [
            "awaiting_clarification",
            PRE_DRILL_KEY,
            AWAITING_PRE_DRILL,
            "pre_drill_nodes",
            "pre_drill_query",
            DOC_CHOICES_KEY,
            "filtered_nodes",
            "hier_sections",
            "clarification_level",
            "policy_auto_select",
            "auto_skipped",
            "current_doc",
            "original_user_question",
            "selected_bu",
            "awaiting_bu_selection",
            "clarification_just_exited",
            "selected_h1",       # ✅ clear H1
            "selected_title",
            "ordered_h2",         # ✅ clear selected section title
            "selected_nodes",         # ✅ clear previously filtered nodes
        ]:
            if key in (
                "awaiting_clarification",
                PRE_DRILL_KEY,
                AWAITING_PRE_DRILL,
                "awaiting_bu_selection",
            ):
                cl.user_session.set(key, False)
            else:
                cl.user_session.set(key, None)

        # Wipe Redis-backed memory
        thread_id = cl.context.session.thread_id
        user_id = cl.user_session.get("user").identifier
        redis_key = f"{user_id}:{thread_id}"
        redis_client.delete(redis_key)

        # Reinitialize memory
        fresh_mem = ChatMemoryBuffer.from_defaults(
            token_limit=TOKEN_LIMIT,
            chat_store=chat_store,
            chat_store_key=redis_key
        )
        cl.user_session.set("memory", fresh_mem)

        # ✅ Inform user in the conversation
        await cl.Message(
            content="🧹 ระบบได้เริ่มต้นการสนทนาใหม่แล้ว กรุณาเลือกหน่วยงานของคุณอีกครั้ง 👇"
        ).send()

        # Trigger BU selection prompt
        logger.info("🔄 User reset triggered → show BU options again")
        await ask_business_unit()
        cl.user_session.set("awaiting_bu_selection", True)

        return

    logger.info(f"Received message from user: {message.content}")
    runnable = cl.user_session.get("runnable")
    retriever = cl.user_session.get("retriever")
    thread_id = cl.context.session.thread_id


    if not runnable or not retriever:
        await send_with_feedback("⚠️ System not ready. Please try again later.")
        return

    # Only set the original user question once
    if cl.user_session.get("original_user_question") is None:
        cl.user_session.set("original_user_question", message.content.strip())

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
        memory = cl.user_session.get("memory")
        # Don't log the number, log the clarified section
        if memory and idx < len(titles):
            clarified_section = titles[idx]
            memory.put(ChatMessage(role="user", content=f"Clarified: {clarified_section}"))
            logger.info(f"✅ Appended clarified section to memory: {clarified_section}")
        # Normal section picked
        selected_title = titles[idx]
        selected_nodes = sections[selected_title]
        logger.info(f"🔍 Hierarchical: user picked “{selected_title}” with {len(selected_nodes)} chunks")

        # ✅ Save clarified user choice to memory
        memory = cl.user_session.get("memory")
        original_q = cl.user_session.get("original_user_question") or message.content
        memory.put(ChatMessage(role="user", content=f"Clarified: {selected_title}"))
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
    
async def show_h1_options(message):
    h1_options = cl.user_session.get("h1_options") or []
    exit_label = "❌ ถามคำถามใหม่"
    opts = h1_options + [exit_label]

    lines = [f"{i+1}. {title}" for i, title in enumerate(opts)]
    text = "❓ โปรดเลือกหัวข้อหลัก (ระดับ 1):\n\n" + "\n".join(lines)
    text += "\n\n🔴 ตอบด้วยหมายเลข หรือพิมพ์ชื่อหัวข้อที่ต้องการหรือพิมพ์ '0' เพื่อเลือก Business Group ใหม่"

    await cl.Message(
        content=text,
        author="Customer Service Agent"
    ).send()


async def handle_standard_query(message: cl.Message):
    """Handles a standard, non-clarification query with hierarchical clarification."""

    import re
    from collections import defaultdict
    from difflib import SequenceMatcher
    import statistics
    # At the top of handle_standard_query
    memory = cl.user_session.get("memory")
    if memory:
        memory.put(ChatMessage(role="user", content=message.content))
        logger.info("🧠 Memory after input:")
        for m in memory.get():
            logger.info(f"{m.role}: {m.content}")
    current_q = message.content.strip()
    prev_q    = cl.user_session.get("pre_drill_query")
    original_q = current_q
    orig_q = current_q

    #cl.user_session.set("original_user_question", current_q)
    if cl.user_session.get("drill_level") == "h1":
        h1_options = cl.user_session.get("h1_options") or []
        user_input = message.content.strip()

        selected_h1 = None
        if user_input.isdigit():
            index = int(user_input) - 1
            if 0 <= index < len(h1_options):
                selected_h1 = h1_options[index]
        else:
            from difflib import get_close_matches
            matches = get_close_matches(user_input, h1_options, n=1, cutoff=0.75)
            if matches:
                selected_h1 = matches[0]

        if selected_h1:
            logger.info("✅ H1 drill selected: %s", selected_h1)
            cl.user_session.set("drill_level", None)
            cl.user_session.set("selected_h1", selected_h1)

            # Immediately continue to H2 selection using the filtered H1 chunks
            doc_nodes = cl.user_session.get("pre_drill_nodes") or []
            filtered_nodes = [
                n for n in doc_nodes
                if (path := n.node.metadata.get("section_path", [])) and len(path) >= 2 and path[0] == selected_h1
            ]

            cl.user_session.set("selected_h1", selected_h1)
            cl.user_session.set("h1_filtered_nodes", filtered_nodes)

            # 🧠 Run the same handler again, now with drill_level cleared and H1 locked
            
            return await handle_standard_query(message)

            
        else:
            logger.warning("❌ Invalid H1 input: %s", user_input)

            # NEW: Heuristic fallback — treat it as new question if it looks like a real sentence
            if len(user_input) > 10 and not user_input.isdigit():
                logger.info("🧠 User likely typed a real question, exiting H1 drill mode.")
                cl.user_session.set("drill_level", None)
                cl.user_session.set("h1_options", None)
                return await handle_standard_query(message)

            await cl.Message("❌ ไม่พบหัวข้อที่คุณเลือก โปรดลองอีกครั้ง หรือพิมพ์ 0 เพื่อเริ่มใหม่").send()
            return

    
     # ─── 11) Fuzzy-fallback & final LLM answer ─────────────────────────────────
    # compute how close we are to any of your canned Q→A
    fuzzy_scores = {
        q: SequenceMatcher(None, current_q.lower(), q.lower()).ratio()
        for q in predefined_answers
    }
    best_q, fuzzy_score = max(fuzzy_scores.items(), key=lambda kv: kv[1], default=("", 0.0))

    if fuzzy_score >= FUZZY_THRESHOLD:
        logger.info(f"✅ Fuzzy override: “{current_q}” ≈ “{best_q}” ({fuzzy_score:.2f}) → predefined answer")
        await send_with_feedback(predefined_answers[best_q], author="Customer Service Agent")
        return
    
        # ─── 4) Prepare retrieval █────────────────────────────────────
    retriever = cl.user_session.get("retriever")
    thread_id = cl.context.session.thread_id
    memory = cl.user_session.get("memory")
    past = memory.get()[-3:]
    context = "\n".join(f"{m.role.title()}: {m.content.strip()}" for m in past if m.content.strip())
    query_with_context = f"{context}\nUser: {message.content}" if context else message.content

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

        # Enforce BU filtering at pre-drill stage
        selected_bu = cl.user_session.get("selected_bu") or "ALL"
        if selected_bu != "ALL":
            allowed_docs = BU_DOCUMENT_MAP.get(selected_bu, [])
            all_nodes = [
                n for n in all_nodes
                if n.node.metadata.get("source", "").split("/")[-1] in allowed_docs
            ]
            logger.info("📁 Filtered doc list for BU=%s → %s", selected_bu, allowed_docs)

        cl.user_session.set("pre_drill_nodes", all_nodes)

        # Log each document’s best score
        doc_scores = {}
        for n in all_nodes:
            src = n.node.metadata.get("source", "Unknown")
            doc_scores[src] = max(doc_scores.get(src, 0.0), n.score)
        for src, score in doc_scores.items():
            logger.info(f"🔍 Doc candidate: '{src}' with top score {score:.3f}")
            
        


        # ─── AUTO-SELECT Policy FAQ.docx if confident ───

        POLICY_AUTO_THRESH    = 0.5   # only policy FAQ ≥0.55 auto-selects
        DOC_CANDIDATE_THRESH  = 0.5   # any doc ≥0.40 is eligible for the user to choose
        BU_RELEVANCE_THRESHOLD = 0.50
        policy_score = doc_scores.get("Policy FAQ.docx", 0.0)
        top_score = max(doc_scores.values(), default=0.0)
        if top_score < BU_RELEVANCE_THRESHOLD:
            logger.warning(
                "❌ Rejected: top_score %.3f is below BU_RELEVANCE_THRESHOLD %.3f → Question may not be relevant to selected BU/doc",
                top_score,
                BU_RELEVANCE_THRESHOLD,
            )
            await cl.Message(
                content=(
                    "คำถามนี้ดูเหมือนไม่เกี่ยวข้องกับเอกสารในหน่วยงานที่คุณเลือกไว้ (BU: **%s**).\n\n"
                    "กรุณาลองเลือกคำถามใหม่ หรือเลือก BU อื่นที่ตรงกับเนื้อหามากกว่า"
                ) % cl.user_session.get("selected_bu", "N/A")
            ).send()
            return  # ⛔ Stop the flow here

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
            # If that yields ≤1 but you still have multiple docs overall, fall back
            if len(doc_set) <= 1 and len(doc_scores) > 1:
                # prompt on the top 5 by score (excluding FAQ)
                doc_set = [
                    src for src, _ in
                    sorted(doc_scores.items(), key=lambda x: -x[1])
                    if src != "Policy FAQ.docx"
                ][:5]
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
        cl.user_session.set("selected_doc", selected_doc)
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

        # Enforce BU filtering again after user selects doc
        selected_bu = cl.user_session.get("selected_bu") or "ALL"
        if selected_bu != "ALL":
            allowed_docs = BU_DOCUMENT_MAP.get(selected_bu, [])
            retrieved_nodes = [
                n for n in retrieved_nodes
                if n.node.metadata.get("source", "").split("/")[-1] in allowed_docs
            ]
            logger.info("📁 (Doc Re-retrieve) Filtered doc list for BU=%s → %s", selected_bu, allowed_docs)

        # Now filter down to just the user-selected document
        filtered = [
            n for n in retrieved_nodes
            if n.node.metadata.get("source") == selected_doc
        ]

        logger.info("📄 User selected doc: %s", selected_doc)


        cl.user_session.set("pre_drill_nodes", filtered)
        cl.user_session.set("filtered_nodes", filtered)
        # message.content remains unchanged so H2/H3 logic fires normally
        
    # ─── 2b) Handle hierarchical clarification selection ───
    if cl.user_session.get("awaiting_clarification"):
        level = cl.user_session.get("clarification_level", 2)
        hier  = cl.user_session.get("hier_sections", {})   # { title: [nodes] }
        choice = message.content.strip()

        # Build the options
        titles     = list(hier.keys())
        exit_label = "❌ ถามคำถามใหม่"
        opts       = titles + [exit_label]

        idx = None

        # 1) Digit?
        if choice.isdigit():
            i = int(choice) - 1
            if 0 <= i < len(opts):
                idx = i

        # 2) Exact title?
        if idx is None and choice in titles:
            idx = titles.index(choice)

        # 3) Fuzzy match (ratio > 0.6)
        if idx is None:
            from difflib import SequenceMatcher
            best = (0.0, None)   # (ratio, index)
            for i, t in enumerate(titles):
                r = SequenceMatcher(None, choice, t).ratio()
                if r > best[0]:
                    best = (r, i)
            if best[0] > 0.6:
                idx = best[1]

        # 4) Exit label
        if idx is None and choice == exit_label:
            idx = len(opts) - 1

        # Invalid?
        if idx is None:
            logger.warning(f"⚠️ Invalid hierarchical choice: {choice}")
            await send_with_feedback("⚠️ โปรดเลือกหมายเลขหรือชื่อหัวข้อให้ถูกต้อง")
            return

        selected = opts[idx]
        logger.info(f"🔍 Hierarchical: user picked “{selected}” at level {level}")

        # Exit → restart flow
        if selected == exit_label:
            cl.user_session.set("clarification_just_exited", True)
            return await handle_standard_query(message)

        # Clear menu flags
        cl.user_session.set("awaiting_clarification", False)
        cl.user_session.set("clarification_level", None)
        
        # ─── NEW: top-level H2 pick → only shortcut if no H3 children ───
        if level == 1:
            # look at pre_drill_nodes to see if there are any H3 under this H2
            all_nodes = cl.user_session.get("h1_filtered_nodes") or cl.user_session.get("pre_drill_nodes") or []
            has_h3 = any(
                len(n.node.metadata.get("section_path", [])) >= 3 and
                n.node.metadata["section_path"][1] == selected
                for n in all_nodes
            )
            if not has_h3:
                # no deeper subdivisions → answer immediately on best H2 chunk
                h2_nodes = hier[selected]
                best_h2_chunk = max(h2_nodes, key=lambda n: n.score)
                logger.info(f"✅ H2 “{selected}” has no H3 → immediate answer (score {best_h2_chunk.score:.3f})")
                clear_clarification_state()
                #orig_q = cl.user_session.get("original_user_question")
                combined_text = "\n\n".join(n.node.text for n in raw_h3[h2_key])
                await send_with_feedback(
                    f"✅ นี่คือสิ่งที่พบจาก “{h2_key}”:\n\n{combined_text}\n\nหากต้องการเริ่มคำถามใหม่ กรุณาพิมพ์ 0",
                    author="Customer Service Agent"
                )
                clear_clarification_state()
                cl.user_session.set("awaiting_clarification", False)
                return
            # otherwise fall through into your existing H3‐menu logic

        # H2 → show H3 menu
        if level == 2:
            # Grab full pre-drill nodes
            all_nodes = cl.user_session.get("h1_filtered_nodes") or cl.user_session.get("pre_drill_nodes") or []
            from collections import defaultdict
            raw_h3 = defaultdict(list)
            for n in all_nodes:
                path = n.node.metadata.get("section_path", [])
                if len(path) >= 3 and path[1] == selected:
                    raw_h3[path[2]].append(n)

            # No H3 → answer on H2
            if not raw_h3:
                matching_chunks = [
                    n for n in all_nodes
                    if len(n.node.metadata.get("section_path", [])) >= 2
                    and n.node.metadata["section_path"][1] == selected
                ]
                clear_clarification_state()
                cl.user_session.set("awaiting_clarification", False)
                #orig_q = cl.user_session.get("original_user_question")

                logger.info("📚 No H3 found under H2: %s → %d chunks sent", selected, len(matching_chunks))
                return await answer_from_node(matching_chunks, orig_q)

            # Otherwise show H3 choices
            cl.user_session.set("awaiting_clarification", True)
            cl.user_session.set("clarification_level", 3)
            cl.user_session.set("hier_sections", { h3: raw_h3[h3] for h3 in raw_h3 })

            opts = list(raw_h3.keys()) + [exit_label]
            lines = [f"{i+1}. {title}" for i, title in enumerate(opts)]
            logger.info(f"🏷 Showing H3 menu with {len(raw_h3)} options")
            await send_with_feedback(
                "❓ โปรดเลือกหัวข้อย่อย (ระดับ 3):\n\n" + "\n".join(lines),
                author="Customer Service Agent"
                
            )
            logger.info("🧠 Current chat memory state: %s", cl.user_session.to_dict())
            return

        # H3 → answer immediately
        else:  # level == 3
            h3_nodes = hier[selected]
            logger.info(f"✅ H3 selected: “{selected}” → {len(h3_nodes)} chunks sent via answer_from_nodes")
            clear_clarification_state()
            cl.user_session.set("awaiting_clarification", False)
            #orig_q = cl.user_session.get("original_user_question")
            return await answer_from_node(h3_nodes, orig_q)



    # ─── 3) Reset on new question ───
    if not cl.user_session.get("awaiting_clarification") and current_q != prev_q:
        cl.user_session.set("auto_skipped", False)
        cl.user_session.set("hier_sections", None)
        cl.user_session.set("clarification_level", None)
        cl.user_session.set("filtered_nodes", None)
    



    # ─── 5) Retrieve (or reuse filtered_nodes) █────────────────────────────────────
    nodes = cl.user_session.get("filtered_nodes")
    if nodes is None:
        try:
            # Retrieve raw nodes
            query = message.content.strip()
            nodes = retriever.retrieve(query)  # <--- You already defined `retriever` earlier in your function

            # ─── Enforce BU filtering early ─────────────────────────────
            selected_bu = cl.user_session.get("selected_bu") or "ALL"
            if selected_bu != "ALL":
                allowed_docs = BU_DOCUMENT_MAP.get(selected_bu, [])
                nodes = [n for n in nodes if n.node.metadata.get("source", "").split("/")[-1] in allowed_docs]
                logger.info("📁 Filtered doc list for BU=%s → %s", selected_bu, allowed_docs)

            if not nodes:
                logger.warning("⚠️ No matching documents found for BU=%s. Falling back to all.", selected_bu)
                await send_with_feedback("ไม่พบเอกสารที่เกี่ยวข้องกับ BU นี้", metadata={"difficulty": "Rejected"})
                return

            # Store for reuse
            cl.user_session.set("filtered_nodes", nodes)

            for i, n in enumerate(nodes[:3], 1):
                snippet = n.node.get_text().strip().replace("\n", " ")
                logger.info(
                    "🏷 Top #%d: source=%s score=%.3f\n    chunk=\"%s\"",
                    i,
                    n.node.metadata.get("source"),
                    n.score,
                    snippet[:200]
                )

            # Fuzzy fallback
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
                    orig_q = cl.user_session.get("original_user_question") or query_with_context
                    return await answer_from_node(best_node, orig_q)

        except Exception:
            logger.exception("❌ Retrieval failed")
            await send_with_feedback("⚠️ เกิดข้อผิดพลาดในการค้นหา กรุณาลองใหม่อีกครั้ง")
            return

    # ─── 6) Scores, early-reject, auto-drill & auto-answer ─────────────────────
    # ─── 6) Scores, early-reject, auto-drill & auto-answer ─────────────────────
    pre_drill_nodes = cl.user_session.get("pre_drill_nodes")
    if pre_drill_nodes:
        logger.info("📦 Overriding nodes with user-selected document chunks (pre_drill_nodes)")
        nodes = pre_drill_nodes

    top_score = nodes[0].score if nodes else 0.0
    logger.info(
        f"🧪 DEBUG | top_score={top_score:.4f}, "
        f"VECTOR_MIN={VECTOR_MIN_THRESHOLD:.4f}, "
        f"VECTOR_MEDIUM={VECTOR_MEDIUM_THRESHOLD:.4f}"
    )
    logger.warning(
        f"🔍 About to check early-reject: top_score={top_score:.4f} vs VECTOR_MIN_THRESHOLD={VECTOR_MIN_THRESHOLD:.4f}"
    )

    # Log top-ranked docs after BU filtering
    logger.info("📑 Top-ranked docs after BU filtering:")
    for n in nodes[:10]:  # log top 10
        doc_name = n.node.metadata.get("source", "").split("/")[-1]
        logger.info("🔍 Doc candidate: '%s' with score %.3f", doc_name, n.score)

    # pick the highest-scoring chunk
    # ─── Promote near-top H2 chunks over H1 ───
    H2_OVERRULE_DELTA = 0.01
    # partition by heading level
    h1_nodes = [n for n in nodes if len(n.node.metadata.get("section_path", [])) == 1]
    h2_nodes = [n for n in nodes if len(n.node.metadata.get("section_path", [])) >= 2]

    if h2_nodes and h1_nodes:
        best_h1_node = max(h1_nodes, key=lambda n: n.score)
        best_h2_node = max(h2_nodes, key=lambda n: n.score)
        if best_h2_node.score >= best_h1_node.score - H2_OVERRULE_DELTA:
            best_node = best_h2_node
        else:
            best_node = best_h1_node
    else:
        best_node = max(nodes, key=lambda n: n.score)

    best_path = best_node.node.metadata.get("section_path", [])
    depth = len(best_path)

    # ─── NEW: deepest‐level + confidence + gap shortcut ───
    DEEP_DIRECT_THRESHOLD = 0.60
    DEEP_GAP_THRESHOLD    = 0.055

    # look at your full pre‐drill to see how deep your document actually goes
    all_pre_drill = cl.user_session.get("pre_drill_nodes") or []
    max_depth    = max(len(n.node.metadata.get("section_path", [])) for n in all_pre_drill)
    target_depth = max_depth - 1

    # only consider when our best_node is at the deepest H3 level
    if depth >= 3 and depth == target_depth and best_node.score >= DEEP_DIRECT_THRESHOLD:
        # extract the H2 under which best_node lives
        h2_key = best_path[1]

        # 1) gather true H3 siblings (same depth, same parent H2)
        sibling_scores = [
            n.score
            for n in all_pre_drill
            if (
                len(n.node.metadata.get("section_path", [])) == depth
                and n.node.metadata["section_path"][1] == h2_key
            )
        ]

        # 2) fallback: if none (weird), include any chunk under that H2
        if not sibling_scores:
            sibling_scores = [
                n.score
                for n in all_pre_drill
                if (
                    len(n.node.metadata.get("section_path", [])) >= 2
                    and n.node.metadata["section_path"][1] == h2_key
                )
            ]

        sibling_scores.sort(reverse=True)
        top       = sibling_scores[0]
        runner_up = sibling_scores[1] if len(sibling_scores) > 1 else 0.0
        gap       = top - runner_up

        logger.info(f"🏷 DEBUG siblings H3 scores under H2 “{h2_key}”: {sibling_scores}")
        logger.info(
            f"🏷 DEBUG deepest‐level check: depth={depth}, top={top:.3f}, "
            f"runner_up={runner_up:.3f}, gap={gap:.3f} (threshold {DEEP_GAP_THRESHOLD})"
        )

        if gap >= DEEP_GAP_THRESHOLD:
            logger.info(f"🏷 Deepest‐level direct‐answer (gap {gap:.3f} ≥ {DEEP_GAP_THRESHOLD})")

            # Use all nodes from the same section_path as the best_node
            section_path = best_node.node.metadata.get("section_path", [])
            all_nodes = cl.user_session.get("h1_filtered_nodes") or cl.user_session.get("pre_drill_nodes") or []
            matching_section = [
                n for n in all_nodes
                if n.node.metadata.get("section_path", []) == section_path
            ]

            clear_clarification_state()
            cl.user_session.set("awaiting_clarification", False)
            #orig_q = cl.user_session.get("original_user_question") or query_with_context
            return await answer_from_node(matching_section, orig_q)
        else:
            logger.info(f"🏷 Gap too small ({gap:.3f} < {DEEP_GAP_THRESHOLD}) → showing H3 menu")
    # ────────────────────────────────────────────────────────

    # (…then falls through into your normal “no-H2s” or “auto‐drill” or H2/H3 menu code…)
    # ─── fallback to H2/H3 menu as before ───

    # ─── Otherwise fall back to your normal H2/H3 menu logic ───

    # 6a) Early‐reject at top levels (depth<3)
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

    # ─── 6b) Auto‐drill into H3 of the highest‐scoring H2 (skip H2 menu) ───────────
    VECTOR_AUTO_LEVEL3_THRESHOLD = 0.6
    if not cl.user_session.get("awaiting_clarification") and top_score >= VECTOR_AUTO_LEVEL3_THRESHOLD:
        from collections import defaultdict
        top_k_nodes = nodes[:5]
        all_doc_nodes = cl.user_session.get("h1_filtered_nodes") or cl.user_session.get("pre_drill_nodes") or []  # ✅ use all nodes for hierarchy

        # group all H2 sections from all_nodes (not just top_k)
        all_nodes = cl.user_session.get("pre_drill_nodes") or []
        raw_h2 = defaultdict(list)
        for n in all_nodes:
            path = n.node.metadata.get("section_path", [])
            if len(path) >= 2:
                raw_h2[path[1]].append(n)

        logger.info(f"🔍 Found {len(raw_h2)} unique H2 candidates from top_k_nodes")

        if len(raw_h2) == 0:
            logger.warning("⚠️ No valid H2 sections found — skipping auto-drill")
        elif len(raw_h2) == 1:
            logger.info("🛑 Only one H2 candidate — skipping auto-drill to avoid flooding LLM")
        else:
            # pick the top‐scoring H2
            section_scores = {
                h2: max(getattr(n, "score", top_score) or top_score for n in grp)
                for h2, grp in raw_h2.items()
            }
            for h2, score in section_scores.items():
                logger.info(f"📊 Section '{h2}' max score: {score:.3f}")
            top_h2, top_h2_score = max(section_scores.items(), key=lambda x: x[1])
            logger.info(f"🔍 Best H2 candidate: '{top_h2}' with score {top_h2_score:.3f}")

            if top_h2_score >= VECTOR_AUTO_LEVEL3_THRESHOLD:
                # collect all H3 under that H2
                h3_chunks = [
                    n for n in all_nodes  # ✅ again use top_k only
                    if len(n.node.metadata.get("section_path", [])) >= 3
                    and n.node.metadata["section_path"][1] == top_h2
                ]

                if h3_chunks:
                    logger.info("🏷 Auto-drill into H3 for '%s' → %d chunks", top_h2, len(h3_chunks))

                    cl.user_session.set("awaiting_clarification", True)
                    cl.user_session.set("clarification_level", 3)

                    # map H3 titles → lists of nodes
                    h3_map = {}
                    for n in h3_chunks:
                        title = n.node.metadata["section_path"][2]
                        h3_map.setdefault(title, []).append(n)
                    cl.user_session.set("hier_sections", h3_map)

                    # send the H3 menu
                    exit_label = "❌ ถามคำถามใหม่"
                    opts = list(h3_map.keys()) + [exit_label]
                    lines = [f"{i+1}. {title}" for i, title in enumerate(opts)]
                    await send_with_feedback(
                        "❓ โปรดเลือกหัวข้อย่อย (ระดับ 3):\n\n" + "\n".join(lines),
                        author="Customer Service Agent"
                    )
                    return

    # 6c) Auto‐answer if extremely confident
    VECTOR_AUTO_DIRECT_THRESHOLD = 0.62 
    if depth >= 2 and top_score >= VECTOR_AUTO_DIRECT_THRESHOLD:
        logger.info(
            "✅ Auto-answer triggered at depth %d (score %.3f)",
            depth, top_score
        )

        # Use all nodes from the same section_path as the best_node
        section_path = best_node.node.metadata.get("section_path", [])
        all_nodes = cl.user_session.get("h1_filtered_nodes") or cl.user_session.get("pre_drill_nodes") or []
        matching_section = [
            n for n in all_nodes
            if n.node.metadata.get("section_path", []) == section_path
        ]

        clear_clarification_state()
        cl.user_session.set("awaiting_clarification", False)
        #orig_q = cl.user_session.get("original_user_question") or query_with_context
        return await answer_from_node(matching_section, orig_q)


    # ─── 8) 0th-drill: hierarchical section drill ─────────────────────────────────
    # 🧠 Use selected_h1 if set (from previous drill), otherwise detect best_h1
    selected_h1 = cl.user_session.get("selected_h1")
    all_doc_nodes = cl.user_session.get("pre_drill_nodes") or []

    if selected_h1:
        logger.info(f"🛑 H1 already selected: '{selected_h1}', filtering all_doc_nodes")
        all_doc_nodes = [
            n for n in all_doc_nodes
            if (path := n.node.metadata.get("section_path", [])) and path[0] == selected_h1
        ]
        best_h1 = selected_h1
    else:
        best_path = best_node.node.metadata.get("section_path", [])
        best_h1 = best_path[0] if len(best_path) >= 1 else None
        if best_h1:
            cl.user_session.set("selected_h1", best_h1)
    # 🔧 FIX: build section_scores (H1 → max score of its chunks)
    section_scores = defaultdict(float)
    for n in all_doc_nodes:
        path = n.node.metadata.get("section_path", [])
        if len(path) >= 1:
            h1 = path[0]
            section_scores[h1] = max(section_scores[h1], getattr(n, "score", 0.0))
    all_h1s = []
    for n in all_doc_nodes:
        path = n.node.metadata.get("section_path", [])
        if len(path) >= 1:
            all_h1s.append(path[0])
    logger.info("📚 Available H1s in pre_drill_nodes: %s", list(set(all_h1s)))
    logger.info(f"📦 pre_drill_nodes fallback → using {len(all_doc_nodes)} nodes")
    
    # Sort H1s by score
    ordered_h1 = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
    logger.info("📊 H1 candidates by score: %s", ordered_h1)

    if len(ordered_h1) >= 2:
        top_h1, top_score = ordered_h1[0]
        second_h1, second_score = ordered_h1[1]
        score_gap = top_score - second_score
        logger.info("🔍 H1 score gap = %.3f", score_gap)

        if score_gap < 0.08:  # not a big gap, means ambiguity
            cl.user_session.set("drill_level", "h1")
            cl.user_session.set("h1_options", [h1 for h1, _ in ordered_h1[:5]])
            cl.user_session.set("pre_drill_query", current_q)
            cl.user_session.set("pre_drill_nodes", all_doc_nodes)
            return await show_h1_options(message)

    # Collect every H2 under that H1, regardless of retrieval score
    raw_h2 = defaultdict(list)
    for n in all_doc_nodes:
        path = n.node.metadata.get("section_path", [])
        if len(path) >= 2 and path[0] == (selected_h1 or best_h1):
            raw_h2[path[1]].append(n)
    logger.info(f"🔍 Built raw_h2 with {len(raw_h2)} H2 sections from best_h1 = '{best_h1}'")
    for h2, grp in raw_h2.items():
        logger.info(f"🔍 raw_h2['{h2}'] → {len(grp)} chunks, top score {max(n.score for n in grp):.3f}")

    # Build your section_scores map (you may still need it later)
    section_scores = {h2: max(n.score for n in grp) for h2, grp in raw_h2.items()}
    for h2, score in section_scores.items():
        logger.info(f"📊 Section '{h2}' max score: {score:.3f}")
    # ─── Always present *all* H2s in document order ─────────────────────────────
    # ─── Always present H1 drill first if not already done ─────────────────────────────
   # 🛑 If user already selected H1, skip H1 clarification and go straight to H2
    selected_h1 = cl.user_session.get("selected_h1")
    all_doc_nodes = cl.user_session.get("pre_drill_nodes") or []  # ensure fallback

    if selected_h1:
        logger.info(f"🛑 H1 already selected: '{selected_h1}', filtering pre_drill_nodes")
        # Keep only chunks under the selected H1
        filtered_nodes = [
            n for n in all_doc_nodes
            if (path := n.node.metadata.get("section_path", [])) and len(path) >= 2 and path[0] == selected_h1
        ]
        all_doc_nodes = filtered_nodes
        # ⬅️ Right after filtering all_doc_nodes under selected H1
        from collections import Counter

        # Recalculate H2 section scores only under selected H1
        h2_counter = Counter()
        for n in all_doc_nodes:
            path = n.node.metadata.get("section_path", [])
            if len(path) >= 2:
                h2_counter[path[1]] += 1

        total = sum(h2_counter.values())
        section_scores = {k: v / total for k, v in h2_counter.items()}
        logger.info(f"📌 Filtered {len(all_doc_nodes)} nodes under selected H1: {selected_h1}")

    else:
        if cl.user_session.get("drill_level") != "h2":
            # Build H1 options
            h1_options = sorted(set(
                path[0] for n in all_doc_nodes
                if (path := n.node.metadata.get("section_path", [])) and len(path) >= 2
            ))

            if not selected_h1 and len(h1_options) > 1:
                logger.info(f"📋 Prompting user to pick H1 from {len(h1_options)} options")
                cl.user_session.set("drill_level", "h1")
                cl.user_session.set("h1_options", h1_options)
                cl.user_session.set("pre_drill_query", current_q)
                cl.user_session.set("pre_drill_nodes", all_doc_nodes)

                lines = [f"{i+1}. {h1}" for i, h1 in enumerate(h1_options)]
                await send_with_feedback(
                    "❓ โปรดเลือกหัวข้อหลัก (ระดับ 1):\n\n" + "\n".join(lines) + "\n\n❌ ถามคำถามใหม่",
                    author="Customer Service Agent"
                )
                return

    # ─── Proceed to H2 clarification ─────────────────────────────
    all_doc_nodes = cl.user_session.get("pre_drill_nodes") or []
    selected_h1 = cl.user_session.get("selected_h1")  # fallback if needed

    # 🔧 Build hierarchy from all_doc_nodes
    hierarchy = defaultdict(lambda: defaultdict(list))
    for n in all_doc_nodes:
        path = n.node.metadata.get("section_path", [])
        if len(path) >= 2:
            h1, h2 = path[0], path[1]
            hierarchy[h1][h2].append(n)

    logger.info(f"📚 Available H1s in pre_drill_nodes: {list(hierarchy.keys())}")

    # 🔧 🧯 Rebuild hierarchy if selected_h1 is missing
    if selected_h1 not in hierarchy:
        logger.warning(f"⚠️ selected_h1 '{selected_h1}' not found in hierarchy — rebuilding from full nodes")
        full_nodes = cl.user_session.get("filtered_nodes") or all_doc_nodes

        hierarchy = defaultdict(lambda: defaultdict(list))
        for n in full_nodes:
            path = n.node.metadata.get("section_path", [])
            if len(path) >= 2:
                h1, h2 = path[0], path[1]
                hierarchy[h1][h2].append(n)

        logger.info(f"🛠 Rebuilt hierarchy: {list(hierarchy.keys())}")

    # 🧯 Fallback if still missing
    if selected_h1 not in hierarchy:
        logger.warning(f"⚠️ selected_h1 '{selected_h1}' not found in hierarchy keys: {list(hierarchy.keys())}")

        # 🧩 Fallback to answering using H1 chunk only
        fallback_chunks = [
            n for n in all_doc_nodes
            if (path := n.node.metadata.get("section_path")) and len(path) >= 1 and path[0] == selected_h1
        ]

        if fallback_chunks:
            logger.info("📤 Answering using fallback H1 chunk only (no H2)")

            # Store user message in memory
            memory = cl.user_session.get("memory")
            if memory:
                memory.put(ChatMessage(role="user", content=f"Clarified: {selected_h1}"))
                logger.info(f"✅ Appended fallback clarified H1: {selected_h1}")

            await answer_from_node(fallback_chunks, message.content)
        else:
            logger.warning("⚠️ No fallback chunks available for selected_h1")
            await cl.Message("⚠️ ไม่พบเนื้อหาในหัวข้อนี้").send()

        return

    # Proceed to H2 clarification
    ordered_h2 = list(hierarchy[selected_h1].keys())
    raw_h2 = hierarchy[selected_h1]

    if len(ordered_h2) > 1:
        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("clarification_level", 1)
        cl.user_session.set("hier_sections", raw_h2)
        cl.user_session.set("pre_drill_query", message.content)
        cl.user_session.set("pre_drill_nodes", all_doc_nodes)

        exit_label = "❌ ถามคำถามใหม่"
        opts = ordered_h2 + [exit_label]
        lines = [f"{i+1}. {h}" for i, h in enumerate(opts)]

        await send_with_feedback(
            f"❓ โปรดเลือกหัวข้อย่อย (ระดับ 2):\n\n" + "\n".join(lines),
            author="Customer Service Agent"
        )
        return
    else:
        logger.info(f"✅ Only one H2 under {selected_h1}, no clarification needed")

    # ─── Exactly one H2 → drill into H3 or fallback ──────────
    if len(ordered_h2) == 1:
        h2_key = ordered_h2[0]
        logger.info("🏷 Single H2 chosen: %s", h2_key)

        # Build raw_h3 from the only H2
        raw_h3 = defaultdict(list)
        for n in all_doc_nodes:
            path = n.node.metadata.get("section_path", [])
            if len(path) >= 3 and path[1] == h2_key:
                raw_h3[path[2]].append(n)

        # If no H3s exist under the H2, return all chunks under H2 directly
        # If no H3s exist under the H2, fallback to sending all chunks under H2 to LLM
        if not raw_h3:
            logger.info("🧩 No H3 found, fallback to all chunks under H2: %s", h2_key)
            h2_nodes = [
                n for n in all_doc_nodes
                if len(n.node.metadata.get("section_path", [])) >= 2
                and n.node.metadata["section_path"][1] == h2_key
            ]
            return await answer_from_node(h2_nodes, orig_q)

        # Otherwise, show all H3s found under that H2
        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("clarification_level", 3)
        cl.user_session.set("hier_sections", {title: raw_h3[title] for title in raw_h3})

        exit_label = "❌ ถามคำถามใหม่"
        opts = list(raw_h3.keys()) + [exit_label]
        lines = [f"{i+1}. {title}" for i, title in enumerate(opts)]
        await send_with_feedback(
            "❓ โปรดเลือกหัวข้อย่อย (ระดับ 3):\n\n" + "\n".join(lines),
            author="Customer Service Agent"
        )
        return

   

    # ─── 12) LLM answer ─────────────────────────────────────────────
    ctx2 = "\n".join(m.content for m in memory.get()[-3:] if m.role == "user")
    final_q = f"{ctx2}\n{message.content}" if ctx2 else message.content
    lvl = "Hard" if top_score >= VECTOR_MEDIUM_THRESHOLD else "Medium"
    await answer_with_llm(nodes, final_q, lvl, top_score, fuzzy_score)

    # ─── 13) Reset for next new question ─────────────────────────────────
    # ─── 13) Reset for next new question ─────────────────────────────────
    clear_clarification_state()

    # 🧹 Clear all related session state
    for key in [
        "awaiting_clarification",
        PRE_DRILL_KEY,
        AWAITING_PRE_DRILL,
        "pre_drill_nodes",
        "pre_drill_query",
        DOC_CHOICES_KEY,
        "filtered_nodes",
        "hier_sections",
        "clarification_level",
        "policy_auto_select",
        "auto_skipped",
        "current_doc",
        "original_user_question",
        "selected_bu",
        "awaiting_bu_selection",
    ]:
        if key in ("awaiting_clarification", PRE_DRILL_KEY, AWAITING_PRE_DRILL, "awaiting_bu_selection"):
            cl.user_session.set(key, False)
        else:
            cl.user_session.set(key, None)

    # 💬 Inform the user in the chat window

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

    # 🔪 Split any long chunk using heading markers like "#", "##"
    split_contexts = []
    for src, txt in contexts:
        sub_chunks = re.split(r"(?=^#+ )", txt, flags=re.MULTILINE)  # split at "#", "##", etc.
        for chunk in sub_chunks:
            clean_chunk = chunk.strip()
            if clean_chunk:
                split_contexts.append((src, clean_chunk))

    # Replace the original contexts
    contexts = split_contexts
    
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
    cl.user_session.set("reset_memory_next_turn", True)


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