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
VECTOR_MIN_THRESHOLD = 0.4
VECTOR_MEDIUM_THRESHOLD = 0.75
CONTEXT_WINDOW = 12000

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
            "model": GROQ_MODEL_ID_2,
            #"api_base": LLM_BASE_URL,
            "api_key": GROQ_API_KEY,
            "is_chat_model": True,
            "is_function_calling_model": False,
            "temperature": 0.2,
            #"http_client": httpx.Client(verify=False),
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

    if chat_profile == "Accounting Compliance 2":
        return OpenAILike(**settings)
    elif chat_profile == "Accounting Compliance":
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
    for key in [
        "awaiting_clarification", "clarification_rounds", "fuzzy_clarification_rounds",
        "possible_summaries", "nodes_to_consider", "summary_to_meta", "original_query"
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
    logger.info(f"Received message from user: {message.content}")
    runnable = cl.user_session.get("runnable")
    retriever = cl.user_session.get("retriever")
    thread_id = cl.context.session.thread_id

    save_conversation_log(thread_id, message.id, role="user", content=message.content)
    memory = cl.user_session.get("memory")
    memory.put(ChatMessage(role="user", content=message.content))
    # Prevent leftover input from being interpreted as a new question


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
        fuzzy_candidates = [(q, s[2]) for q, s in summary_to_meta.items() if isinstance(s, tuple) and s[0] == "fuzzy"]

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

        # Nothing to do
        await send_with_feedback("⚠️ ไม่พบข้อมูลที่เกี่ยวข้อง กรุณาพิมพ์คำถามใหม่")
        clear_clarification_state()
        return

    cl.user_session.set("clarification_rounds", rounds + 1)
    choice = message.content.strip()
    opt_out_index = None
    opt_out_label = "❌ ถามคำถามใหม่"
    if opt_out_label in summaries:
        opt_out_index = summaries.index(opt_out_label)

    # If user types a number that matches the opt-out
    if choice.isdigit() and opt_out_index is not None and int(choice) - 1 == opt_out_index:
        clear_clarification_state()
        cl.user_session.set("awaiting_clarification", False)
        cl.user_session.set("clarification_just_exited", True)
        await send_with_feedback("✅ คุณได้เลือกเริ่มต้นคำถามใหม่ กรุณาถามคำถามของคุณอีกครั้ง")
        return

    # Also still support typing the opt-out text manually
    if choice.strip() in ["❌", "❌ ถามคำถามใหม่", "exit", "ถามคำถามใหม่"]:
        clear_clarification_state()
        cl.user_session.set("awaiting_clarification", False)
        cl.user_session.set("clarification_just_exited", True)
        await send_with_feedback("✅ คุณได้เลือกเริ่มต้นคำถามใหม่ กรุณาถามคำถามของคุณอีกครั้ง")
        return
        clear_clarification_state()
        cl.user_session.set("awaiting_clarification", False)
        cl.user_session.set("clarification_just_exited", True)
        cl.user_session.set("clarification_reset_waiting", True)  # ✅ NEW
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
    summary_to_meta = cl.user_session.get("summary_to_meta", {})
    fuzzy_rounds = cl.user_session.get("fuzzy_clarification_rounds", 0)

    if selected_index >= len(summaries):
        await send_with_feedback("⚠️ ไม่พบหัวข้อที่เลือก โปรดพิมพ์หมายเลขหรือชื่อหัวข้อให้ถูกต้องอีกครั้ง")
        return

    selected_summary = summaries[selected_index]

    # ✅ Auto-pick fuzzy match if too many clarification rounds
    if fuzzy_rounds >= MAX_FUZZY_CLARIFICATION_ROUNDS:
        fuzzy_candidates = [(q, s[2]) for q, s in summary_to_meta.items() if isinstance(s, tuple) and s[0] == "fuzzy"]
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

    # ✅ If user selected a fuzzy entry
    if isinstance(summary_to_meta.get(selected_summary), tuple) and summary_to_meta[selected_summary][0] == "fuzzy":
        _, answer, score = summary_to_meta[selected_summary]
        await send_with_feedback(
            f"{answer}\n\n🧠 *DEBUG* | Category: **Easy (Clarified)** | Method: **Predefined** | Fuzzy: {score:.2f}",
            metadata={"difficulty": "Easy"},
        )
        save_conversation_log(cl.context.session.thread_id, None, "bot", answer, "Easy")
        clear_clarification_state()
        return

    # ✅ Else fallback to vector node
    if nodes_to_consider and selected_index < len(nodes_to_consider):
        chosen_node = nodes_to_consider[selected_index]
        await answer_from_node(chosen_node, original_query)
    elif nodes_to_consider:
        chosen_node = max(nodes_to_consider, key=lambda n: n.score)
        await answer_from_node(chosen_node, original_query)
        clear_clarification_state()
    else:
        await send_with_feedback("⚠️ ไม่พบเนื้อหาที่เกี่ยวข้อง โปรดลองเลือกหัวข้อใหม่หรือลองถามใหม่อีกครั้ง")
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
    """Handles a standard, non-clarification query."""
    # Short-circuit: ignore leftover responses after exiting clarification
    # If clarification just ended, reset and wait for a real new query
    if cl.user_session.get("clarification_reset_waiting"):
        content = message.content.strip()
        if content.isdigit() or len(content) <= 2:
            await send_with_feedback("✅ กรุณาพิมพ์คำถามใหม่ที่คุณต้องการสอบถาม")
            return
        cl.user_session.set("clarification_reset_waiting", False)

    if cl.user_session.get("clarification_just_exited"):
        cl.user_session.set("clarification_just_exited", False)
        cl.user_session.set("clarification_reset_waiting", True)
        cl.user_session.set("awaiting_clarification", False)
        cl.user_session.set("clarification_rounds", 0)
        cl.user_session.set("fuzzy_clarification_rounds", 0)
        cl.user_session.set("possible_summaries", None)
        cl.user_session.set("nodes_to_consider", None)
        cl.user_session.set("summary_to_meta", None)
        cl.user_session.set("original_query", None)
        
        return

    # ✅ Reset prior clarification state
    cl.user_session.set("awaiting_clarification", False)
    cl.user_session.set("clarification_rounds", 0)
    cl.user_session.set("fuzzy_clarification_rounds", 0)
    cl.user_session.set("possible_summaries", None)
    cl.user_session.set("nodes_to_consider", None)
    cl.user_session.set("summary_to_meta", None)
    cl.user_session.set("original_query", None)

    retriever = cl.user_session.get("retriever")
    runnable = cl.user_session.get("runnable")
    thread_id = cl.context.session.thread_id
    cl.user_session.set("fuzzy_clarification_rounds", 0)

    # Step 1: Fuzzy match
    fuzzy_scores = {q: SequenceMatcher(None, message.content.lower(), q.lower()).ratio() for q in predefined_answers}
    fuzzy_candidates = sorted(
        [(q, s) for q, s in fuzzy_scores.items() if s > FUZZY_THRESHOLD],
        key=lambda x: x[1],
        reverse=True
    )
    best_question, fuzzy_score = fuzzy_candidates[0] if fuzzy_candidates else ("", 0)
    best_match = predefined_answers.get(best_question, "")

    # Step 2: Vector retrieval
# Step 2: Vector retrieval
    try:
        query_with_context = message.content
        memory = cl.user_session.get("memory")
        past_messages = memory.get()[-3:]  # Include last 3 turns (user + assistant)
        context_snippets = "\n".join(f"{m.role.title()}: {m.content.strip()}" for m in past_messages if m.content.strip())
        if context_snippets:
            query_with_context = context_snippets + "\nUser: " + message.content

        nodes = retriever.retrieve(query_with_context)
        top_score = nodes[0].score if nodes else 0.0
        logger.info("📥 Retrieved chunks from Vector DB:")
        for i, n in enumerate(nodes):
            preview = n.node.text[:120].replace("\n", " ")
            logger.info(f"  {i+1}. Title: {n.node.metadata.get('section_title', 'Unknown')} | Score: {n.score:.4f} | Preview: {preview}")
    except Exception as e:
        await send_with_feedback(f"⚠️ Retrieval error: {str(e)}")
        return

    # Step 3: Return predefined answer if it's stronger
    if fuzzy_score > top_score:
        if len(fuzzy_candidates) > 1:
            logger.info("🔍 Multiple fuzzy matches above threshold. Triggering clarification.")
            return await start_clarification_flow(nodes=[], original_query=message.content, fuzzy_candidates=fuzzy_candidates)
        level = "Easy"
        content = (
            f"{best_match}\n\n"
            f"🧠 *DEBUG* | Category: **Easy** | Method: **Predefined** | "
            f"Fuzzy: {fuzzy_score:.2f} | Vector: {top_score:.2f} | Matched Q: {best_question}"
        )
        await send_with_feedback(content, metadata={"easy": level})
        save_conversation_log(thread_id, message.id, "bot", best_match, level)
        return

    # Step 3: Handle low-confidence fallback and ambiguity checks
    import statistics
    scores = [n.score for n in nodes[:MAX_TOPICS_BEFORE_CLARIFY]]

    if fuzzy_score < FUZZY_THRESHOLD and (not scores or scores[0] < VECTOR_MIN_THRESHOLD):
        cl.user_session.set("clarification_just_exited", True)
        cl.user_session.set("clarification_reset_waiting", True)
        cl.user_session.set("awaiting_clarification", False)
        await send_with_feedback("❗ยังไม่สามารถหาคำตอบได้จากคำถามนี้ กรุณาระบุคำถามให้ชัดเจนขึ้น เช่น อ้างอิงหัวข้อหรือประเภทค่าใช้จ่าย")
        return

    vector_ambiguous = (
        len(scores) >= 3 and scores[0] >= 0.45 and statistics.stdev(scores[:3]) < 0.007
    )
    fuzzy_ambiguous = len(fuzzy_candidates) > 1
    if vector_ambiguous or fuzzy_ambiguous:
        return await start_clarification_flow(nodes, message.content, fuzzy_candidates)

    # Step 5: LLM-based answer
    query_with_context = message.content
    memory = cl.user_session.get("memory")
    past_messages = memory.get()[-3:]  # Use last 3 user messages as context
    context_snippets = "\n".join(m.content for m in past_messages if m.role == "user")
    if context_snippets:
        query_with_context = context_snippets + "\n" + message.content

    if top_score >= VECTOR_MEDIUM_THRESHOLD:
        level = "Hard"
        await answer_with_llm(nodes, query_with_context, level, top_score, fuzzy_score)
    elif top_score >= VECTOR_MIN_THRESHOLD:
        level = "Medium"
        await answer_with_llm(nodes, query_with_context, level, top_score, fuzzy_score)
    else:
        level = "Rejected"
        content = (
            "❌ คำถามนี้ไม่เกี่ยวข้องกับนโยบาย LOA / DoA กรุณาถามในหัวข้อที่เกี่ยวข้อง\n\n"
            f"🧠 *DEBUG* | Category: **Rejected** | Vector: {top_score:.2f} | Fuzzy: {fuzzy_score:.2f}"
        )
        await send_with_feedback(content, metadata={"difficulty": level})
        save_conversation_log(thread_id, message.id, "bot", "Rejected", level)

async def start_clarification_flow(nodes: list, original_query: str, fuzzy_candidates: list = None):
    """Initiates the clarification process when a query is too broad."""
    # Ensure fuzzy_clarification_rounds is initialized
    cl.user_session.set("clarification_just_exited", False)
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
    """Generates an answer using the LLM with context from retrieved nodes."""
    runnable = cl.user_session.get("runnable")
    TOP_K = 3
    selected_nodes = nodes[:TOP_K]
    contexts = [(n.node.metadata.get("source", "Unknown"), n.node.text.strip().replace("\n", " ")) for n in selected_nodes]

    # Include top vector chunks
    chunk_context = "".join(
        f'({i}) เอกสารนโยบาย: "{src}"\nเนื้อหาชิ้นนี้ (เต็มข้อความ):\n"""{txt}\n"""\n\n'
        for i, (src, txt) in enumerate(contexts, 1)
    )

    # Include prior messages (from memory)
    memory: ChatMemoryBuffer = cl.user_session.get("memory")
    

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

    # Compose final context
    context_str = (
        f"📜 ประวัติการสนทนา:\n{history_snippets}\n\n"
        f"📚 ข้อความจากเอกสาร:\n{chunk_context}"
    )

    # Dynamically detect if response should be table-like
    suggest_table = any(
        "20 ล้านบาท" in txt or "500,000 บาท" in txt or "ประเภทที่" in txt or "โครงการ" in txt
        for _, txt in contexts
    )

    formatting_hint = ""

    if suggest_table:
        formatting_hint = (
            "\n🔶 คำแนะนำสำคัญ: กรุณาสรุปคำตอบในรูปแบบ **ตาราง Markdown** โดยระบุหัวข้อหลักเช่น 'ประเภทโครงการ', 'มูลค่า', 'เอกสารที่ต้องจัดทำ', และ 'ผู้มีอำนาจอนุมัติ' "
            "หากไม่สามารถจัดตารางได้ ให้สรุปในรูปแบบ bullet โดยอิงเนื้อหาด้านล่าง"
        )
    else:
        formatting_hint = "\nหากเหมาะสม ให้จัดคำตอบในรูปแบบ bullet หรือย่อหน้าเพื่อความเข้าใจง่าย"

    filtered_query = (
        f'ผู้ใช้ถามว่า: "{query}"\n\n'
        f"กรุณาตอบโดยอาศัยเนื้อหาต่อไปนี้ทั้งหมด:\n\n{context_str}"
        f"{formatting_hint}\nในคำตอบให้ระบุชื่อเอกสารนโยบายที่ใช้ และชี้ว่าอ้างอิงมาจากข้อความใดบ้าง"
    )

    try:
        resp = runnable.query(filtered_query)
        answer_body = resp.response.strip() if hasattr(resp, "response") else "".join(resp.response_gen).strip()
    except Exception as e:
        answer_body = f"⚠️ LLM error: {e}"

    answer_body = extract_and_format_table(answer_body)
    sources_used = ", ".join(f'"{src}"' for src, _ in contexts)
# ✅ Render clean markdown table without code block
    if "|" in answer_body and "---" in answer_body:
        answer_body = f"**คำตอบ**\n\n{answer_body.strip()}"

    final_answer = (
        f"{answer_body}\n\n"
        f"🧠 *DEBUG* | Category: **{level}** | Method: **VectorStore + LLM (top {TOP_K})** | Vector: {top_score:.2f} | Fuzzy: {fuzzy_score:.2f}"
    )
    memory.put(ChatMessage(role="assistant", content=answer_body))
    await send_with_feedback(final_answer, metadata={"difficulty": level})
    save_conversation_log(cl.context.session.thread_id, cl.context.session.id, "bot", final_answer, level)

    # Detect if user query is about a form
    form_keywords = ["ฟอร์ม", "แบบฟอร์ม", "form", "แนบ", "กรอก", "template", "request form"]
    query_is_form_related = any(k in query.lower() for k in form_keywords)

    # Search for links in the top-k contexts
    form_links = set()
    for n in selected_nodes:
        link = n.node.metadata.get("attachment_link")
        if link:
            form_links.add(link)

    # Append form link section if applicable
    form_links_text = ""
    if query_is_form_related and form_links:
        form_links_text = "\n\n📎 แนบลิงก์แบบฟอร์มหรือเอกสารที่เกี่ยวข้อง:\n" + "\n".join(f"- {url}" for url in form_links)
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