# Run application locally using this command: chainlit run app.py -h --root-path /chatbot/v1
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.groq import Groq
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage

from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference

from llama_index.core.prompts import PromptTemplate
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from functions.qdrant_vectordb import QdrantManager

from chainlit.types import ThreadDict
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
from typing import Dict, Optional
from dotenv import load_dotenv
from pathlib import Path
import chainlit as cl
import os
import time
import logging
import warnings
import json
import difflib
from difflib import SequenceMatcher
import requests
import redis
import asyncio
import urllib.parse
import hashlib
from bs4 import BeautifulSoup
import markdown
import re

import chainlit as cl

from chainlit import Action
from fastapi import Request
import httpx
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from sqlalchemy import Table, Column, String, JSON, MetaData
from sqlalchemy.ext.asyncio import AsyncEngine
from patches import patch
patch.apply_patch()




# metadata for our extra table
extra_meta = MetaData()

clarification_state = Table(
    "clarification_state",
    extra_meta,
    Column("thread_id", String, primary_key=True),
    Column("summaries", JSON, nullable=False),
    Column("nodes", JSON, nullable=False),  # you'll serialize node.score, node.text, node.metadata
)




MAX_CLARIFICATION_ROUNDS = 2
MAX_TOPICS_BEFORE_CLARIFY = 3
# Parse from env
parsed = urllib.parse.urlparse(os.getenv("REDIS_CHATSTORE_URI"))
redis_client = redis.Redis(
    host=parsed.hostname,
    port=parsed.port or 6379,
    password=os.getenv("REDIS_CHATSTORE_PASSWORD"),
    db=0
)
warnings.filterwarnings("ignore")


# Load from root .env file (one level up from chainlit_app)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
from prompts import SYSTEM_PROMPT_STANDARD, SYSTEM_PROMPT_DEEPTHINK

print("📡 MS_TEAMS_WORKFLOW_URL:", os.getenv("MS_TEAMS_WORKFLOW_URL"))

print("✅ Loaded CHAINLIT_AUTH_SECRET:", os.getenv("CHAINLIT_AUTH_SECRET"))
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load variables from the .env file
load_dotenv()
LLM_MODEL_ID    = os.getenv("LLM_MODEL_ID")
LLM_BASE_URL    = os.getenv("LLM_BASE_URL")
API_KEY_CHATBOT = os.getenv("API_KEY_CHATBOT")
EMBED_MODEL_ID  = os.getenv("EMBED_MODEL_ID")
EMBED_BASE_URL  = os.getenv("EMBED_BASE_URL")
# Access the variables
GROQ_MODEL_ID_1 = os.getenv("GROQ_MODEL_ID_1")
GROQ_MODEL_ID_2 = os.getenv("GROQ_MODEL_ID_2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Collection Name (Vector database)
QDANT_COLLENCTION_NAME = os.getenv("QDANT_COLLENCTION_NAME")

# Chat-Memory
REDIS_CHATSTORE_URI = os.getenv("REDIS_CHATSTORE_URI")
REDIS_CHATSTORE_PASSWORD = os.getenv("REDIS_CHATSTORE_PASSWORD")
# Chat-Memory Token limit
TOKEN_LIMIT = 512

# Phoenix
TRACE_ENDPOINT = os.getenv("TRACE_ENDPOINT")
TRACE_PROJECT_NAME = os.getenv("TRACE_PROJECT_NAME")

tracer_provider = register(
    project_name=TRACE_PROJECT_NAME,
    endpoint=TRACE_ENDPOINT,
    set_global_tracer_provider=False,
)

LlamaIndexInstrumentor().instrument(
    skip_dep_check=True, tracer_provider=tracer_provider
)


chat_store = RedisChatStore(
    redis_url=REDIS_CHATSTORE_URI, db=0, password=REDIS_CHATSTORE_PASSWORD, ttl=180
)

# Initialize QdrantManager
qdrant_manager = QdrantManager()

# Set the desired chunk size and context window
# Settings.chunk_size = 512



from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

@cl.data_layer
def get_data_layer():
    # Use the ASYNC_DATABASE_URL, pointing to the asyncpg driver
    return SQLAlchemyDataLayer(conninfo=os.environ["ASYNC_DATABASE_URL"])




# 1) Generic recorder for any reaction-like action
import chainlit as cl
import time, json, logging

logger = logging.getLogger(__name__)


# …and similarly for multi-star emojis if you choose…








def get_current_chainlit_thread_id() -> str:
    return cl.context.session.thread_id

# Constants and configurations
DATASET_MAPPING = {
    "Standard": QDANT_COLLENCTION_NAME,
    "Deepthink": QDANT_COLLENCTION_NAME,
    "Accounting Compliance": QDANT_COLLENCTION_NAME,
}

CHAT_ENGINE_PARAMS = {
    'chat_mode': "context",
    'similarity_top_k': 5,
    'sparse_top_k': 12,
    'alpha': 0.5,
    'vector_store_query_mode': 'hybrid'
}

CHAT_PROFILES = {
    "Standard": {
        "context_prompt": SYSTEM_PROMPT_STANDARD,
        "welcome_message": "Hello {firstname}, how can i help you today?",
        "llm_settings": {
            "model": GROQ_MODEL_ID_1,
            "api_key": GROQ_API_KEY,
            "is_chat_model": True,
            "is_function_calling_model": False,
            "temperature": 0.7,
        },
    },
    "Deepthink": {
        "context_prompt": SYSTEM_PROMPT_DEEPTHINK,
        "welcome_message": "Hello {firstname}, how can i help you today?",
        "llm_settings": {
            "model": GROQ_MODEL_ID_2,
            "api_key": GROQ_API_KEY,
            "is_chat_model": True,
            "is_function_calling_model": False,
            "temperature": 0.7,
        },
    },
        "Accounting Compliance": {  # ✅ ADD THIS
        "context_prompt": SYSTEM_PROMPT_STANDARD,
        "welcome_message": "Hi there! Need help with accounting compliance?",
        "llm_settings": {
            "model": LLM_MODEL_ID,
            "api_base": LLM_BASE_URL,
            "api_key": API_KEY_CHATBOT,
            "is_chat_model": True,
            "is_function_calling_model": False,
            "temperature": 0.2,
            "http_client": httpx.Client(verify=False),
        },
    },
}


def get_llm_settings(chat_profile: str):
    cfg = CHAT_PROFILES.get(chat_profile, {}).get("llm_settings")
    if not cfg:
        raise ValueError(f"No settings for {chat_profile}")
    return Groq(
        model=cfg["model"],
        api_key=cfg["api_key"],
        is_chat_model=cfg["is_chat_model"],
        is_function_calling_model=cfg["is_function_calling_model"],
        temperature=cfg["temperature"],
    )


async def answer_from_node(node, user_q):
    """Builds and sends the final LLM response from a single node."""
    src = node.node.metadata.get("source","Unknown")
    txt = node.node.text.replace("\n"," ")
    prompt = (
        f"ผู้ใช้ถามว่า: \"{user_q}\"\n"
        f"บทความที่เลือกมาจากเอกสารนโยบาย \"{src}\" (เต็มข้อความ):\n\"\"\"\n{txt}\n\"\"\"\n\n"
        "กรุณาตอบโดยอาศัยเนื้อหาในบทความนี้"
    )
    resp = cl.user_session.get("runnable").query(prompt)
    answer = resp.response if hasattr(resp, "response") else "".join(resp.response_gen)
    answer = extract_and_format_table(answer.strip())
    # reset state
    for k in ["awaiting_clarification","clarification_rounds","possible_summaries","nodes_to_consider","summary_to_meta","original_query"]:
        cl.user_session.set(k, None)
    msg = cl.Message(
        content="",
        metadata={"difficulty": "Clarified"}
    )
    await msg.send()

    stream_text = f"✅ นี่คือสิ่งที่พบจาก “{src}”:\n\n{answer}"
    for char in stream_text:
        await msg.stream_token(char)
        await asyncio.sleep(0.005)  # Optional: adjust speed

    await msg.update()
    save_conversation_log(cl.context.session.thread_id, None, "bot", answer, difficulty="Clarified")
    return

def markdown_table_to_html(md_text: str) -> str:
    """Convert markdown tables to HTML tables, preserving non-table content."""
    if '|' not in md_text or '---' not in md_text:
        return md_text  # Not a markdown table

    html = markdown.markdown(md_text, extensions=['markdown.extensions.tables'])
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')

    if table:
        table['style'] = (
            "border-collapse: collapse; width: 100%; font-size: 14px; "
            "margin-top: 10px; border: 1px solid #ccc;"
        )
        for th in table.find_all("th"):
            th['style'] = "background-color: #f2f2f2; padding: 8px; border: 1px solid #ccc;"
        for td in table.find_all("td"):
            td['style'] = "padding: 8px; border: 1px solid #ccc;"
        return str(table)

    return md_text

def load_context_prompt(chat_profile: str) -> str:
    """Load the context prompt for the given chat profile."""
    return CHAT_PROFILES.get(chat_profile, {}).get("context_prompt", "")

def save_conversation_log(thread_id: str, parent_id: str, role: str, content: str, difficulty: str = None):
    key = f"conversation_log:{thread_id}"
    log_entry = {
        "timestamp": time.time(),
        "parent_id": parent_id,
        "role": role,  # "user", "bot", or "admin"
        "content": content,
    }
    if difficulty:
        log_entry["difficulty"] = difficulty

    # Append to list stored in Redis
    # Load existing log (if any)
    existing_raw = redis_client.get(key)
    log_list = json.loads(existing_raw) if existing_raw else []

    # Append new entry
    log_list.append(log_entry)

    # Save back entire list as JSON string (overwrite)
    redis_client.set(key, json.dumps(log_list))
    print(f"📝 Logged {role} message to {key}")





Settings.context_window = 6000

# Set up embedding model
Settings.embed_model = CohereEmbedding(
    api_key=os.getenv("COHEAR_API_KEY"),
    model_name=os.getenv("COHEAR_MODEL_ID"),
    input_type="search_document",
    embedding_type="float",
)


# Set up embedding model
Settings.embed_model = TextEmbeddingsInference(
    model_name=EMBED_MODEL_ID,
    base_url=EMBED_BASE_URL,
    auth_token=f"Bearer {API_KEY_CHATBOT}",
    timeout=60,
    embed_batch_size=10,
)

def start_clarification(thread_id: str, topics: list[str]):
    """
    Mark in the user_session that we are awaiting a clarification. 
    Store the list of `topics` (e.g. doc names or node summaries).
    """
    cl.user_session.set("awaiting_clarification", True)
    # Save the actual topics array. You can use whatever strings make sense,
    # e.g. ["Vendor onboarding", "Payment schedule", "Invoice policy", …]
    cl.user_session.set("possible_topics", topics)
    # We also want to remember the original question, so we know how to re-query or index.
    cl.user_session.set("original_query", thread_id)

def clear_clarification():
    cl.user_session.set("awaiting_clarification", False)
    cl.user_session.set("possible_topics", None)
    cl.user_session.set("original_query", None)



def send_to_ms_teams_workflow(question: str, user_email: str, thread_id: str, parent_id: str):
    webhook_url = os.getenv("MS_TEAMS_WORKFLOW_URL")
    formatted_question = f"""{question}

[thread_id:{thread_id}]
[parent_id:{parent_id}]"""

    payload = {
        "question": formatted_question,
        "user": user_email
    }

    print("📤 Sending to MS Teams:", payload)
    try:
        response = requests.post(webhook_url, json=payload)
        print("📬 Response status:", response.status_code)
        print("📬 Response body:", response.text)
    except Exception as e:
        print("❌ Exception sending to MS Teams:", e)


from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine

def create_chat_engine(chat_profile: str, memory: ChatMemoryBuffer):
    # Load system prompt and dataset for this profile
    context_prompt = load_context_prompt(chat_profile)
    dataset = DATASET_MAPPING.get(chat_profile)
    if not dataset:
        logger.error(f"No dataset configured for profile: {chat_profile}")
        return None, None

    # Fetch the Qdrant vector store
    vector_store = qdrant_manager.get_vector_store(dataset, hybrid=True)
    if not vector_store:
        logger.error(f"❌ Failed to get vector store for dataset: {dataset}")
        return None, None

    # Build the Llama-Index
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Configure the LLM
    llm = get_llm_settings(chat_profile)

    # Create a hybrid retriever + LLM query engine
    query_engine = index.as_query_engine(
        retriever_mode="hybrid",
        llm=llm,
        streaming=True,
        verbose=True,
        similarity_top_k=5,
        sparse_top_k=12,
        alpha=0.5,
    )

    # Also expose a standalone retriever if you need raw node access
    retriever = index.as_retriever(similarity_top_k=5)

    return query_engine, retriever

def setup_runnable():
    """Set up the chat engine (runnable) and retriever in the user session."""
    try:
        chat_profile = cl.user_session.get("chat_profile")
        memory = cl.user_session.get("memory")

        if not chat_profile:
            logger.error("chat_profile not found in user session.")
            return
        if not memory:
            logger.error("memory not found in user session.")
            return

        # Configure and set the LLM
        llm = get_llm_settings(chat_profile)
        Settings.llm = llm

        # Build the chat engine + retriever
        chat_engine, retriever = create_chat_engine(chat_profile, memory)
        if chat_engine and retriever:
            cl.user_session.set("runnable", chat_engine)
            cl.user_session.set("retriever", retriever)
        else:
            logger.warning("Failed to create chat engine or retriever.")

    except Exception as e:
        logger.exception("Error setting up runnable: %s", e)

# Mock test authentication
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    print("⚡️ auth_callback triggered")
    print(f"🔐 Username entered: {username}")
    print(f"🔐 Password entered: {password}")

    if (username, password) == ("admin", "admin"):
        print("✅ Login success")
        return cl.User(
            identifier="admin",
            metadata={"role": "ADMIN", "email": "chatbot_admin@gmail.com", "provider": "credentials"}
        )
    
    print("❌ Login failed")
    return None

@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    return [
        cl.ChatProfile(
            name="Accounting Compliance",
            markdown_description="Got questions about the policy? I'm all ears and ready to help you out—just ask!",
            icon="/public/cp_accountant.png",
        ),
        cl.ChatProfile(
            name="Deepthink",
            markdown_description="Powered by deepseek-r1-distill-llama-70b (Groq) model.",
            icon="/public/deepseek-color.png",
        ),
    ]


# Function that sets four starters for welcome screen
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="อำนาจอนุมัติการลงทุน investment project แต่ละประเภท แต่ละมูลค่า",
            message="อำนาจอนุมัติการลงทุน investment project แต่ละประเภท แต่ละมูลค่า",
            icon="/public/search.svg",
            ),
        cl.Starter(
            label="เอกสารที่ต้องใช้สำหรับการเปิด new vendor code มีอะไรบ้าง ?",
            message="เอกสารที่ต้องใช้สำหรับการเปิด new vendor code มีอะไรบ้าง ?",
            icon="/public/search.svg",
            ),
        cl.Starter(
            label="รอบการทำเบิกเงินทดรองจ่าย และการจ่ายเงิน",
            message="รอบการทำเบิกเงินทดรองจ่าย และการจ่ายเงิน",
            icon="/public/search.svg",
            ),
        cl.Starter(
            label="เมื่อไรต้องเปิด PR ผ่านระบบ เมื่อไรสามารถใช้ PO manual (PO กระดาษได้)",
            message="เมื่อไรต้องเปิด PR ผ่านระบบ เมื่อไรสามารถใช้ PO manual (PO กระดาษได้)",
            icon="/public/search.svg",
            )
        ]


import uuid
import asyncio
from sqlalchemy import MetaData, Table, Column
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, insert as pg_insert

@cl.on_chat_start
async def on_chat_start():
    print("💬 on_chat_start called")
    print("👤 user:", cl.user_session.get("user"))

    # 1) Persist the thread row so steps FKs will succeed
    thread_id = cl.context.session.thread_id
    dl: SQLAlchemyDataLayer = get_data_layer()
    engine = dl.engine

    # Define 'threads' with a UUID primary key
    meta = MetaData()
    threads_table = Table(
        "threads",
        meta,
        Column("id", PG_UUID(as_uuid=True), primary_key=True),
    )

    # Convert the string into a real UUID so it binds correctly
    thread_uuid = uuid.UUID(thread_id)

    async with engine.begin() as conn:
        await conn.execute(
            pg_insert(threads_table)
            .values(id=thread_uuid)
            .on_conflict_do_nothing()
        )

    # 2) Everything else as before…
    app_user = cl.user_session.get("user")
    user_email = app_user.metadata.get("email", "Unknown")
    logger.info(f"User {user_email} has started new chat session!!")

    redis_session_id = f"{app_user.identifier}:{thread_id}"
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=TOKEN_LIMIT,
        chat_store=chat_store,
        chat_store_key=redis_session_id,
    )
    cl.user_session.set("memory", memory)

    setup_runnable()

    print(f"🚀 Starting poll_all_admin_replies for thread: {thread_id}")
    asyncio.create_task(poll_all_admin_replies(thread_id))

def strip_html(html: str) -> str:
    return re.sub('<[^<]+?>', '', html).strip()

# Track shown replies and parent messages
shown_admin_reply_ids = {}  # { redis_key: set(reply_ids) }
shown_parent_keys = set()   # Keys where parent question has been shown

import re

def strip_html(html: str) -> str:
    return re.sub('<[^<]+?>', '', html).strip()


# === Define this at the top of your file or before `on_message()` ===
import re

def extract_and_format_table(text: str) -> str:
    """
    Detect contiguous Markdown table blocks (including separator lines of '-' or '|')
    and reformat them into neat, aligned tables. Non-table text is left untouched.
    """
    lines = text.splitlines()
    output_lines = []
    buffer = []

    def flush_table():
        nonlocal buffer, output_lines
        # Split into cells, drop pure-separator rows
        rows = [
            re.split(r"\s*\|\s*", row.strip("| "))
            for row in buffer
            if row.strip() and not re.fullmatch(r"[\|\-\s]+", row)
        ]
        if not rows:
            buffer = []
            return

        # Pad all rows to the same column count
        max_cols = max(len(r) for r in rows)
        for r in rows:
            r += [""] * (max_cols - len(r))

        # Compute each column’s max width
        widths = [max(len(r[i]) for r in rows) for i in range(max_cols)]

        # Build header + separator
        header = "| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(max_cols)) + " |"
        sep    = "|" + "|".join("-" * (widths[i] + 2) for i in range(max_cols)) + "|"
        output_lines.append(header)
        output_lines.append(sep)

        # Build remaining rows
        for row in rows[1:]:
            line = "| " + " | ".join(row[i].ljust(widths[i]) for i in range(max_cols)) + " |"
            output_lines.append(line)

        buffer = []

    for line in lines:
        # If it’s a pipe-line *or* a pure-dash line, keep buffering
        if "|" in line or re.fullmatch(r"[\|\-\s]+", line):
            buffer.append(line)
        else:
            if buffer:
                flush_table()
            output_lines.append(line)

    # Flush any trailing table
    if buffer:
        flush_table()

    return "\n".join(output_lines)

def clean_parent_content(raw_html: str) -> str:
    """Strip HTML tags and remove metadata lines."""
    text = strip_html(raw_html)
    lines = text.splitlines()
    filtered_lines = [
        line for line in lines
        if not any(tag in line for tag in ["[thread_id:", "[parent_id:", "Email:"])
    ]
    return "\n".join(filtered_lines).strip()

shown_admin_replies = {}  # Keeps track of the latest reply ID per parent
def auto_format_markdown_table(text: str) -> str:
    lines = text.strip().split("\n")
    table_lines = []
    normal_lines = []
    is_in_table = False

    for line in lines:
        if "|" in line:
            table_lines.append(line.strip())
            is_in_table = True
        else:
            if is_in_table:
                break  # Stop at the first non-table line after table
            normal_lines.append(line)

    # Split into rows and clean
    rows = [re.split(r"\s*\|\s*", row.strip("| ")) for row in table_lines]
    if not rows:
        return text

    max_cols = max(len(row) for row in rows)
    # Pad rows to have same number of columns
    rows = [row + [""] * (max_cols - len(row)) for row in rows]

    # Pad columns
    col_widths = [max(len(row[i]) for row in rows) for i in range(max_cols)]

    def format_row(row):
        return "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"

    formatted = []
    for i, row in enumerate(rows):
        formatted.append(format_row(row))
        if i == 0:  # header separator
            formatted.append("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")

    return "\n".join(normal_lines + [""] + formatted)

async def poll_all_admin_replies(thread_id: str):
    printed_keys = set()  # Tracks if we've printed parent message once

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

                # Skip if reply already shown
                if shown_admin_replies.get(key_str) == last_reply_id:
                    continue

                # ✅ Only print parent question once
                if key_str not in printed_keys:
                    await send_with_feedback(
                        content=f"🧾 Original Question:\n\n{clean_parent_content(parent_content)}",
                        author="User"
                    )
                    printed_keys.add(key_str)

                # ✅ Push all replies (cleaned)
                for r in replies:
                    reply_id = r.get("id")
                    if not reply_id:
                        continue
                    if shown_admin_replies.get(f"{key_str}:{reply_id}"):
                        continue

                    content = r.get("body", {}).get("content", "")
                    cleaned = strip_html(content)
                    if cleaned and not shown_admin_replies.get(f"{key_str}:{r['id']}"):
                        await send_with_feedback(
                            content=f"📬 Reply from Admin:\n\n{cleaned}",
                            author="Admin",
                            parent_id=parent_id
                        )
                        shown_admin_replies[f"{key_str}:{r['id']}"] = True

                # ✅ Mark the last reply ID for overall key
                if last_reply_id:
                    shown_admin_replies[key_str] = last_reply_id

        except Exception as e:
            print(f"❌ Redis polling error: {e}")

        await asyncio.sleep(5)
        

# What to do when chat is resumed from chat history
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    thread_id = thread.get("id")
    app_user = cl.user_session.get("user")
    redis_session_id = f"{app_user.identifier}:{thread_id}"

    # 1) Rebuild your ChatMemoryBuffer as before
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=TOKEN_LIMIT,
        chat_store=chat_store,
        chat_store_key=redis_session_id,
    )

    root_messages = [
        m for m in thread["steps"]
        if m["parentId"] is None and m.get("output", "").strip()
    ]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.put(ChatMessage(role="user", content=message["output"]))
        else:
            memory.put(ChatMessage(role="assistant", content=message["output"]))

    cl.user_session.set("memory", memory)

    # 2) Load any saved clarification state from Postgres
    dl = get_data_layer()         # your SQLAlchemyDataLayer
    engine = dl.engine            # AsyncEngine
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(clarification_state)
            .where(clarification_state.c.thread_id == thread_id)
        )
        row = result.mappings().first()

    if row:
        data = dict(row) 
        # flag that we’re awaiting clarification
        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("possible_summaries", data["summaries"])

        # rebuild minimal node objects
        from types import SimpleNamespace
        nodes = []
        for n in data["nodes"]:
            nodes.append(SimpleNamespace(
                score=n["score"],
                node=SimpleNamespace(text=n["text"], metadata=n["meta"])
            ))
        cl.user_session.set("nodes_to_consider", nodes)

        # if you serialize summary→meta in the same table, restore it here too:
        # cl.user_session.set("summary_to_meta", data["summary_to_meta"])
        # cl.user_session.set("original_query", data["original_query"])

    setup_runnable()


async def send_with_feedback(content: str, author: str = "Customer Service Agent", parent_id: str = None):
    """
    Send a chat message without any star-rating actions.
    """
    msg = cl.Message(
        content="",
        author=author,
        parent_id=parent_id
    )
    await msg.send()

    for char in content:
        await msg.stream_token(char)
        await asyncio.sleep(0.005)  # Optional delay for animation effect

    await msg.update()
# === LOAD PREDEFINED ANSWERS ===
with open("predefined_answers.json", "r", encoding="utf-8") as f:
    PREDEFINED_ANSWERS = json.load(f)

def fuzzy_match(user_question: str):
    best_match = None
    best_ratio = 0
    for q, answer in PREDEFINED_ANSWERS.items():
        ratio = SequenceMatcher(None, user_question.lower(), q.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = answer
    return best_match if best_ratio >= RELEVANCE_THRESHOLD_EASY else None


# Load predefined answers once
with open("predefined_answers.json", "r", encoding="utf-8") as f:
    predefined_answers = json.load(f)

# … (above imports and setup) …
from chainlit import Action
async def streaming_message_builder(content: str):
    """Stream a full message with animation, character by character."""
    msg = cl.Message(content="")
    await msg.send()

    for char in content:
        await msg.stream_token(char)
        await asyncio.sleep(0.005)  # adjust for desired animation speed

    await msg.update()
    return msg

@cl.on_message
async def on_message(message: cl.Message):
    response = cl.Message(content="")
    await response.send()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",  # or "llama3-8b-8192"
        "messages": [{"role": "user", "content": message.content}],
        "stream": True
    }

    

    runnable = cl.user_session.get("runnable")
    retriever = cl.user_session.get("retriever")
    chat_profile = cl.user_session.get("chat_profile")
    thread_id = cl.context.session.thread_id

    # Log user message
    save_conversation_log(thread_id, message.id, role="user", content=message.content)
    if not runnable or not retriever:
        await send_with_feedback("⚠️ System not ready. Please try again later.")
        return

    # === Clarification Mode Check ===

    awaiting = cl.user_session.get("awaiting_clarification")
    if awaiting:
        nodes_to_consider = cl.user_session.get("nodes_to_consider", [])
        summaries       = cl.user_session.get("possible_summaries", [])
        summary_to_meta = cl.user_session.get("summary_to_meta", {})
        original_query  = cl.user_session.get("original_query", "")
        logger.info(f"[clarify] Enter round; nodes={len(nodes_to_consider)}, summaries={len(summaries)}")

        # ─── track how many times we've asked ───
        rounds = cl.user_session.get("clarification_rounds") or 0
        logger.info(f"[clarify] previously asked {rounds} times")
        if rounds >= MAX_CLARIFICATION_ROUNDS:
            # auto-pick the top node and exit clarification
            chosen = max(nodes_to_consider, key=lambda n: n.score)
            logger.info(f"[clarify] max rounds reached, auto-selecting node with score {chosen.score:.2f}")

            # clear any leftover clarification state
            for k in ["awaiting_clarification","clarification_rounds",
                      "possible_summaries","nodes_to_consider",
                      "summary_to_meta","original_query"]:
                cl.user_session.set(k, None)

            return await answer_from_node(chosen, original_query)

        cl.user_session.set("clarification_rounds", rounds + 1)

        # … now proceed to match the user’s choice …
        choice = message.content.strip()
        logger.info(f"[clarify] User choice: {choice!r}")
        # ✅ Allow user to exit clarification
        if choice.strip() in ["❌", "❌ ถามคำถามใหม่", "exit", "ถามคำถามใหม่"]:
            for k in ["awaiting_clarification", "clarification_rounds", "possible_summaries", "nodes_to_consider", "summary_to_meta", "original_query"]:
                cl.user_session.set(k, None)
            await send_with_feedback("✅ คุณได้เลือกเริ่มต้นคำถามใหม่ กรุณาถามคำถามของคุณอีกครั้ง")
            return

        # ← Insert “auto” bailout here:
        if choice.lower() == "auto":
            chosen = max(nodes_to_consider, key=lambda n: n.score)
            logger.info(f"[clarify] user requested auto, selecting {chosen.score:.2f}")
             # clear state

            # clear any leftover clarification state
            for k in ["awaiting_clarification","clarification_rounds",
                      "possible_summaries","nodes_to_consider",
                      "summary_to_meta","original_query"]:
                cl.user_session.set(k, None)

            return await answer_from_node(chosen, original_query)

        selected_index = None
        SIMILARITY_TIE_THRESHOLD   = 0.05
        # 1) Numeric choice?
        # 1) Numeric choice or opt-out by index
        opt_out_choice = "❌ ถามคำถามใหม่"
        selected_index = None

        if choice.isdigit():
            idx = int(choice) - 1

            # ✅ Case: user selected opt-out by number
            if idx < len(summaries) and summaries[idx] == opt_out_choice:
                for k in ["awaiting_clarification", "clarification_rounds", "possible_summaries", "nodes_to_consider", "summary_to_meta", "original_query"]:
                    cl.user_session.set(k, None)
                await send_with_feedback("✅ คุณได้เลือกเริ่มต้นคำถามใหม่ กรุณาพิมพ์คำถามของคุณอีกครั้ง")
                return

            # ✅ Case: normal numeric selection
            if 0 <= idx < len(nodes_to_consider):
                selected_index = idx
            else:
                print(f"[DEBUG] number out of range: {idx} / {len(nodes_to_consider)}")

        else:
            # Fallback: fuzzy match against summaries
            best_ratio = 0.0
            best_i = None
            for i, s in enumerate(summaries):
                r = SequenceMatcher(None, choice.lower(), s.lower()).ratio()
                if r > best_ratio:
                    best_ratio = r
                    best_i = i

            if best_ratio > 0.6:
                # ✅ Check again for opt-out by name
                if summaries[best_i] == opt_out_choice:
                    for k in ["awaiting_clarification", "clarification_rounds", "possible_summaries", "nodes_to_consider", "summary_to_meta", "original_query"]:
                        cl.user_session.set(k, None)
                    await send_with_feedback("✅ คุณได้เลือกเริ่มต้นคำถามใหม่ กรุณาพิมพ์คำถามของคุณอีกครั้ง")
                    return

                selected_index = best_i
            else:
                print(f"[DEBUG] fuzzy fail ⇒ best_ratio={best_ratio:.2f}")

        if selected_index is None:
            await send_with_feedback(
                "⚠️ ไม่พบหัวข้อที่เลือก โปรดพิมพ์หมายเลขหรือชื่อหัวข้อให้ถูกต้องอีกครั้ง"
            )
            return

        # At this point, one node is chosen; now do your tie-break logic…
        # At this point, one node is chosen; now do your tie-break logic…
                # Pick the node (with a fallback if it somehow ended up None)
        chosen_node = nodes_to_consider[selected_index]
        if chosen_node is None:
            # Try recovering from summary_to_meta
            summary = summaries[selected_index]
            meta = summary_to_meta.get(summary)
            # meta might be (node_obj, truncated, src) or just node_obj
            if isinstance(meta, tuple):
                chosen_node = meta[0]
            else:
                chosen_node = meta
            if chosen_node is None:
                await send_with_feedback("⚠️ เกิดข้อผิดพลาดภายใน โปรดลองอีกครั้ง")
                return

        # Now only include real nodes in your tie‐break
        tied = [
            n for n in nodes_to_consider
            if n is not None and abs(n.score - chosen_node.score) < SIMILARITY_TIE_THRESHOLD
        ]
        from llama_index.core.llms import ChatMessage

        # 1) If still multiple tied, re-summarize and ask again:
        if len(tied) > 1:
            llm = get_llm_settings(chat_profile)
            new_summaries = []
            new_meta = {}
            for n in tied:
                # truncate and prepare prompt
                full = n.node.text.replace("\n", " ")
                trunc = full if len(full) <= 1000 else full[:1000] + "…"
                prompt = (
                    f"ผู้ใช้ถามว่า: \"{original_query}\"\n"
                    f"เนื้อหาในเอกสารนี้ (เต็มข้อความ):\n\"\"\"\n{trunc}\n\"\"\"\n"
                    "กรุณาสรุปเนื้อหานี้เป็นหัวข้อสั้น ๆ (ไม่เกิน 10 คำ) "
                    "เพื่อให้ผู้ใช้เลือกหัวข้ออีกครั้ง"
                )
                resp = llm.chat([ChatMessage(role="user", content=prompt)])
                title = resp.message.content.strip().split("\n")[0]
                if title not in new_meta:
                    new_meta[title] = n
                    new_summaries.append(title)

            # store for the next round
            # ✅ Add opt-out option
            opt_out_choice = "❌ ถามคำถามใหม่"
            if opt_out_choice not in new_summaries:
                new_summaries.append(opt_out_choice)
            cl.user_session.set("nodes_to_consider", tied)
            cl.user_session.set("possible_summaries", new_summaries)
            cl.user_session.set("summary_to_meta", new_meta)

            await send_with_feedback(
                content=(
                    "❓ ยังพบเนื้อหาหลายรายการที่คะแนนใกล้เคียงกัน กรุณาช่วยระบุหัวข้อเพิ่มเติม\n\n"
                    + "\n".join(f"{i+1}. {s}" for i, s in enumerate(new_summaries))
                    + "\n\nโปรดตอบกลับด้วยหมายเลขหรือชื่อหัวข้อที่ต้องการ หรือเลือก \"❌ ถามคำถามใหม่\" หากต้องการเริ่มต้นใหม่"
                ),
                author="Customer Service Agent"
            )
            return

        # 2) Exactly one remains → clear state & answer:
        for k in ["awaiting_clarification","clarification_rounds",
                "possible_summaries","nodes_to_consider",
                "summary_to_meta","original_query"]:
            cl.user_session.set(k, None)
        return await answer_from_node(chosen_node, original_query)

        # … else you would loop back into summarization of `tied` and re-ask …
        # At this point, “selected_index” is valid:
        chosen_node = nodes_to_consider[selected_index]

    # … (rest of your clarification logic) …

        # 7) Check if any other node is still “tied” in score
        SIMILARITY_TIE_THRESHOLD = 0.05
        chosen_score = chosen_node.score
        tied_nodes = []
        for lbl, (node_obj, _, _) in summary_to_meta.items():
            if abs(node_obj.score - chosen_score) < SIMILARITY_TIE_THRESHOLD:
                tied_nodes.append(node_obj)
        logger.info(f"[clarify] chosen_score={chosen_score:.2f}, tied_nodes={len(tied_nodes)}")
        # ─── prevent endless loops if LLM returns the same list ───
        prev = set(cl.user_session.get("possible_summaries") or [])
        if prev and set(summaries) == prev:
            # nothing changed—auto-resolve
            chosen = max(nodes_to_consider, key=lambda n: n.score)
            cl.user_session.set("awaiting_clarification", False)
            cl.user_session.set("clarification_rounds", None)
            return await answer_from_node(chosen, original_query)

        # 8) If more than one node remains tied, re‐summarize just those
        if len(tied_nodes) > 1:
            logger.info("[clarify] still ambiguous, regenerating summaries for tied nodes")
            llm = get_llm_settings(chat_profile)
            new_summaries = []
            new_summary_to_meta = {}

            for n in tied_nodes:
                full_text = n.node.text.replace("\n", " ")
                truncated = full_text if len(full_text) <= 1000 else full_text[:1000] + "…"
                src = n.node.metadata.get("source", "UnknownPolicy")

                summary_prompt = (
                    f"ผู้ใช้ถามว่า: \"{original_query}\"\n"
                    f"เนื้อหาในเอกสารนี้ (เต็มข้อความ):\n\"\"\"\n{truncated}\n\"\"\"\n"
                    "กรุณาสรุปเนื้อหานี้เป็นหัวข้อสั้น ๆ (ไม่เกิน 10 คำ) "
                    "เพื่อให้ผู้ใช้เลือกหัวข้ออีกครั้ง"
                )
                print("-----[DEBUG] (Re‐summarize tied) Full node text:", truncated)
                try:
                    from llama_index.core.llms import ChatMessage
                    resp = llm.chat([ChatMessage(role="user", content=summary_prompt)])
                    one_line = resp.message.content.strip().split("\n")[0].strip()
                except Exception:
                    one_line = truncated[:40].strip() + ("…" if len(truncated) > 40 else "")

                if one_line not in new_summary_to_meta:
                    new_summary_to_meta[one_line] = (n, truncated, src)
                    new_summaries.append(one_line)
            # ✅ Add opt-out option
            opt_out_choice = "❌ ถามคำถามใหม่"
            if opt_out_choice not in new_summaries:
                new_summaries.append(opt_out_choice)
            # Overwrite the session with this new “still-tied” group
            cl.user_session.set("possible_summaries", new_summaries)
            cl.user_session.set("summary_to_meta", new_summary_to_meta)

            await send_with_feedback(
                content=(
                    "❓ ยังพบเนื้อหาหลายรายการที่คะแนนใกล้เคียงกัน กรุณาช่วยระบุหัวข้อเพิ่มเติม\n\n"
                    "หัวข้อที่เป็นไปได้:\n"
                    + "\n".join(f"{i+1}. {s}" for i, s in enumerate(new_summaries))
                    + "\n\nโปรดตอบกลับด้วยหมายเลขหรือชื่อหัวข้อที่ต้องการ หรือเลือก \"❌ ถามคำถามใหม่\" หากต้องการเริ่มต้นใหม่"
                ),
                author="Customer Service Agent"
            )
            logger.info(f"[clarify] prompt round {rounds+1} sent with {len(new_summaries)} options")
            return

        # 9) Exactly one node remains → build the final LLM prompt
        logger.info(f"[clarify] resolved to single node {chosen_node.score:.2f}, exiting loop")
        cl.user_session.set("awaiting_clarification", False)
        cl.user_session.set("possible_summaries", None)
        cl.user_session.set("summary_to_meta", None)
        cl.user_session.set("original_query", None)

        filtered_query = (
            f"ผู้ใช้ถามว่า: \"{original_query}\"\n"
            f"บทความที่เลือกมาจากเอกสารนโยบาย \"{source_name}\" (เต็มข้อความ):\n"
            f"\"\"\"\n{truncated_text}\n\"\"\"\n\n"
            "กรุณาตอบโดยอาศัยเนื้อหาในบทความนี้ และในคำตอบให้ระบุชื่อเอกสารนโยบายด้วย"
        )
        print("-----[DEBUG] Final filtered_query (single node):-----")
        print(filtered_query)

        try:
            resp = runnable.query(filtered_query)
            if hasattr(resp, "response"):
                answer = resp.response.strip()
            else:
                tokens = [tok for tok in resp.response_gen]
                answer = "".join(tokens).strip()
        except Exception as e:
            answer = f"⚠️ LLM error: {e}"
            print(f"-----[DEBUG] Exception in filtered LLM call: {e}-----")

        answer = extract_and_format_table(answer)

        # 1) Build your star-rating buttons
        buttons = [
            Action(name=f"feedback_{i}", label="⭐" * i, payload={})
            for i in range(1, 6)
        ]

        # 2) Send your answer + buttons in one call
        msg = cl.Message(
            content="",
            actions=buttons
        )
        await msg.send()

        full_text = (
            f"✅ นี่คือสิ่งที่พบจากเอกสารนโยบาย “{source_name}”:\n\n"
            f"{answer}\n\n"
            f"---\n"
            f"🧠 *DEBUG* Filtered on document: {source_name}"
        )

        for char in full_text:
            await msg.stream_token(char)
            await asyncio.sleep(0.005)  # Adjust speed if needed

        await msg.update()

        # 3) Persist your log
        save_conversation_log(thread_id, message.id, role="bot", content=answer, difficulty="Clarified")

        return


    # === Thresholds ===
    FUZZY_THRESHOLD = 0.7
    VECTOR_MIN_THRESHOLD = 0.45
    VECTOR_MEDIUM_THRESHOLD = 0.75

    # === Step 1: Fuzzy match ===
    best_match = None
    best_question = None
    fuzzy_score = 0.0
    for q, a in predefined_answers.items():
        score = SequenceMatcher(None, message.content.lower(), q.lower()).ratio()
        if score > fuzzy_score:
            fuzzy_score = score
            best_question = q
            best_match = a

    # === Step 2: Vector retrieval ===
    try:
        nodes = retriever.retrieve(message.content)
        for i, node in enumerate(nodes):
            print(f"📄 Source {i+1} | score: {node.score:.2f}")
            print(node.node.text)
            print("-" * 60)
        top_score = nodes[0].score if nodes and nodes[0].score is not None else 0.0
    except Exception as e:
        await send_with_feedback(f"⚠️ Retrieval error: {str(e)}")
        return

    # === Ambiguity (Too-broad) Check ===
    # === Ambiguity (Too-broad) Check ===
    # === Ambiguity (Too‐broad) Check ===
    # === Ambiguity (Too‐broad) Check ===
    # === Ambiguity (Too‐broad) Check ===
    # === SIMILARITY_TIE_THRESHOLD = 0.03
    # === MAX_TOPICS_BEFORE_CLARIFY = 3
    SIMILARITY_TIE_THRESHOLD   = 0.05
    scores = [n.score for n in nodes[:MAX_TOPICS_BEFORE_CLARIFY]]
    if (
        len(scores) >= 2
        and all(
            abs(nodes[i].score - nodes[0].score) < SIMILARITY_TIE_THRESHOLD
            for i in range(1, len(scores))
        )
        and nodes[0].score >= VECTOR_MIN_THRESHOLD
    ):
        # Filter out any node that doesn't contain at least one token from the original question
        original_query = message.content
        original_tokens = re.findall(r"\w+", original_query)

        filtered_nodes = []
        for n in nodes[:MAX_TOPICS_BEFORE_CLARIFY]:
            text = n.node.text.replace("\n", " ")
            if any(tok in text for tok in original_tokens):
                filtered_nodes.append(n)

        if len(filtered_nodes) >= 2:
            nodes_to_summarize = filtered_nodes
        else:
            nodes_to_summarize = nodes[:MAX_TOPICS_BEFORE_CLARIFY]

        # Summarize each node into a one‐line title
        summaries = []
        summary_to_meta = {}  # now stores (truncated_text, source_filename)
        llm = get_llm_settings(chat_profile)

        for n in nodes_to_summarize:
            # 1) Extract full text and truncate
            full_text = n.node.text.replace("\n", " ")
            truncated = full_text if len(full_text) <= 1000 else full_text[:1000] + "…"

            # 2) Pull source filename from metadata
            source_name = n.node.metadata.get("source", "UnknownPolicy")

            summary_prompt = (
                f"ผู้ใช้ถามว่า: \"{message.content}\"\n"
                f"เนื้อหาในเอกสารนี้ (เต็มข้อความ):\n\"\"\"\n{truncated}\n\"\"\"\n"
                "กรุณาสรุปเนื้อหานี้เป็นหัวข้อเพื่อให้ผู้ใช้เลือก นี่คือหัวข้อย่อยที่ผู้ใช้จะเลือกเมื่อคำตอบอาจเป็นไปได้หลายกรณี เริ่มต้นประโยคด้วย กรณี เสมอ"
            )

            print("-----[DEBUG] Full node text being summarized:-----")
            print(truncated)
            print("-----[DEBUG] Prompt sent to LLM for summary:-----")
            print(summary_prompt)

            from llama_index.core.llms import ChatMessage
            try:
                resp = llm.chat([ChatMessage(role="user", content=summary_prompt)])
                one_line = resp.message.content.strip().split("\n")[0].strip()
                print("-----[DEBUG] LLM returned one-line summary:-----")
                print(one_line)
            except Exception as e:
                one_line = truncated[:40].strip() + ("…" if len(truncated) > 40 else "")
                print("-----[DEBUG] LLM call failed; using fallback snippet:-----")
                print(one_line)
                print(f"-----[DEBUG] Exception: {e}-----")

            if one_line not in summary_to_meta:
                summaries.append(one_line)
                # Store both truncated text and source filename
                summary_to_meta[one_line] = (n, truncated, source_name)

        # If only one summary, auto-select it
        if len(summaries) == 1:
            selected_topic = summaries[0]
            # Clear clarification state
            cl.user_session.set("possible_summaries", summaries)
            cl.user_session.set("nodes_to_consider", nodes_to_summarize)
            cl.user_session.set("summary_to_meta", summary_to_meta)
            cl.user_session.set("original_query", message.content)
            cl.user_session.set("awaiting_clarification", True)

            from sqlalchemy.dialects.postgresql import insert
            from sqlalchemy.ext.asyncio import AsyncSession

            # Build your payload
            payload = {
                "summaries": summaries,
                "nodes": [
                    {"score": n.score,
                    "text":  n.node.text,
                    "meta":  n.node.metadata}
                    for n in nodes_to_summarize
                ],
                # if you want to restore summary_to_meta/original_query later, include them here too
                "summary_to_meta": summary_to_meta,
                "original_query": original_query,
            }

            # Persist (upsert) into your clarification_state table
            dl = get_data_layer()           # SQLAlchemyDataLayer
            engine = dl.engine              # its AsyncEngine
            async with AsyncSession(engine) as session:
                await session.execute(
                    insert(clarification_state)
                    .values(thread_id=thread_id, **payload)
                    .on_conflict_do_update(
                        index_elements=["thread_id"],
                        set_=payload
                    )
                )
                await session.execute(stmt)
                await session.commit()

            truncated_text, source_name = summary_to_meta[selected_topic]
            filtered_query = (
                f"ผู้ใช้ถามว่า: \"{original_query}\"\n"
                f"บทความที่เลือกมาจากเอกสารนโยบาย \"{source_name}\" "
                f"ภายใต้หัวข้อ \"{selected_topic}\" (เต็มข้อความ):\n"
                f"\"\"\"\n{truncated_text}\n\"\"\"\n"
                "กรุณาตอบโดยอาศัยเนื้อหาในบทความนี้ และในคำตอบให้ระบุชื่อเอกสารนโยบายและหัวข้อด้วย"
            )

            print("-----[DEBUG] Single‐choice auto‐selected topic:-----", selected_topic)
            print("-----[DEBUG] Filtered query sent to LLM:-----")
            print(filtered_query)

            from llama_index.core.llms import ChatMessage
            try:
                resp = runnable.query(filtered_query)
                if hasattr(resp, "response"):
                    answer = resp.response.strip()
                else:
                    tokens = [token for token in resp.response_gen]
                    answer = "".join(tokens).strip()
                print("-----[DEBUG] Final LLM answer:-----")
                print(answer)
            except Exception as e:
                answer = f"⚠️ LLM error: {e}"
                print(f"-----[DEBUG] Exception in filtered LLM call: {e}-----")

            answer = extract_and_format_table(answer)
            await send_with_feedback(
                content=(
                    f"✅ นี่คือสิ่งที่พบจาก “{selected_topic}” ในเอกสารนโยบาย “{source_name}”:\n\n"
                    f"{answer}\n\n"
                    f"---\n"
                    f"🧠 *DEBUG* Auto‐selected single topic"
                ),
                author="Customer Service Agent",
                metadata={"difficulty": "Clarified"}
            )
            save_conversation_log(thread_id, message.id, role="bot", content=answer, difficulty="Clarified")
            return

        # Otherwise, show the list so the user can choose:
        # ✅ Add opt-out option before displaying
        opt_out_choice = "❌ ถามคำถามใหม่"
        if opt_out_choice not in summaries:
            summaries.append(opt_out_choice)
            cl.user_session.set("possible_summaries", summaries)

        await send_with_feedback(
            content=(
                "❓ พบเอกสารหลายรายการที่อาจเกี่ยวข้องกับคำถามของคุณ\n\n"
                "หัวข้อที่เป็นไปได้:\n"
                + "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
                + "\n\nโปรดตอบกลับด้วยหมายเลขหรือชื่อหัวข้อที่ต้องการ หรือเลือก \"❌ ถามคำถามใหม่\" หากต้องการเริ่มต้นใหม่"
            ),
            author="Customer Service Agent"
        )

        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("possible_summaries", summaries)       # ← store under “possible_summaries”
        cl.user_session.set("nodes_to_consider", nodes_to_summarize)  # ← store the actual Node objects
        cl.user_session.set("summary_to_meta", summary_to_meta)
        cl.user_session.set("original_query", message.content)

        from sqlalchemy.dialects.postgresql import insert
        from sqlalchemy.ext.asyncio import AsyncSession

        # after your cl.user_session.set(...) calls:
        dl = get_data_layer()                # your SQLAlchemyDataLayer instance
        engine = dl.engine                   # AsyncEngine
        # serialize only the bits you need
        payload = {
            "summaries": summaries,
            "nodes": [
                {"score": n.score,
                "text":  n.node.text,
                "meta":  n.node.metadata}
                for n in nodes_to_summarize
            ],
        }
        async with AsyncSession(engine) as session:
            await session.execute(
                insert(clarification_state)
                .values(thread_id=thread_id, **payload)
                .on_conflict_do_update(
                    index_elements=["thread_id"],
                    set_=payload
                )
            )
            await session.commit()
        return

    # === Step 4: Compute “level” based on vector score ===
    if top_score >= VECTOR_MEDIUM_THRESHOLD:
        level = "Hard"
    elif top_score >= VECTOR_MIN_THRESHOLD:
        level = "Medium"
    else:
        level = "Rejected"

    if level == "Rejected":
        content = (
            "❌ คำถามนี้ไม่เกี่ยวข้องกับนโยบาย LOA / DoA กรุณาถามในหัวข้อที่เกี่ยวข้อง\n\n"
            "---\n"
            "🧠 *DEBUG*\n"
            "Category: **Rejected**\n"
            f"Vector Score: {top_score:.2f}\n"
            f"Fuzzy Score: {fuzzy_score:.2f}"
        )

        msg = cl.Message(
            content="",
            author="Customer Service Agent",
            metadata={"difficulty": "Rejected"}
        )
        await msg.send()

        for char in content:
            await msg.stream_token(char)
            await asyncio.sleep(0.005)  # Optional animation delay

        await msg.update()
        save_conversation_log(thread_id, message.id, role="bot", content="Rejected", difficulty="Reject")
        return

    if level == "Easy":
        # Predefined answer path (fuzzy override or no vector)
        await send_with_feedback(
            content=(
                f"{best_match}\n\n"
                "---\n"
                "🧠 *DEBUG*\n"
                "Category: **Easy**\n"
                "Answer Method: **Predefined**\n"
                f"Vector Score: {top_score:.2f}\n"
                f"Fuzzy Score: {fuzzy_score:.2f}\n"
                f"Matched Q: {best_question}"
            ),
            author="Customer Service Agent",
            metadata={"difficulty": "Easy"}
        )
        save_conversation_log(thread_id, message.id, role="bot", content=best_match, difficulty="Easy")
        return

    if level == "Medium":
        # ── Hybrid VectorStore + LLM path: send top 3 chunks into the LLM ──
        TOP_K = 3
        selected_nodes = nodes[:TOP_K]

        # 1) Extract (source, text) for each of the top K nodes
        contexts = []
        for n in selected_nodes:
            src = n.node.metadata.get("source", "UnknownPolicy")
            txt = n.node.text.strip().replace("\n", " ")
            contexts.append((src, txt))

        # 2) Build a combined “context_str” that lists each chunk with its source
        context_str = ""
        for idx, (src, txt) in enumerate(contexts, start=1):
            context_str += (
                f"({idx}) เอกสารนโยบาย: \"{src}\"\n"
                f"เนื้อหาชิ้นนี้ (เต็มข้อความ):\n\"\"\"\n{txt}\n\"\"\"\n\n"
            )

        # 3) Craft a single “filtered_query” that includes all top‐K chunks
        filtered_query = (
            f"ผู้ใช้ถามว่า: \"{message.content}\"\n\n"
            f"กรุณาตอบโดยอาศัยเนื้อหาต่อไปนี้ทั้งหมด:\n\n"
            f"{context_str}"
            "ในคำตอบให้ระบุชื่อเอกสารนโยบายที่ใช้ และชี้ว่าอ้างอิงมาจากข้อความใดบ้าง"
        )

        print("-----[DEBUG] Context sent to LLM (top 3 chunks):-----")
        print(context_str)
        print("-----[DEBUG] Filtered query sent to LLM (Medium, multi‐chunk):-----")
        print(filtered_query)

        # 4) Query the LLM with this combined context
        try:
            resp = runnable.query(filtered_query)
            if hasattr(resp, "response"):
                answer_body = resp.response.strip()
                print("===== [DEBUG] runnable.query(...) returned (response): =====")
                print(answer_body)
            else:
                tokens = [token for token in resp.response_gen]
                answer_body = "".join(tokens).strip()
                print("===== [DEBUG] runnable.query(...) returned (streaming): =====")
                print(answer_body)
        except Exception as e:
            answer_body = f"⚠️ LLM error: {e}"
            print(f"-----[DEBUG] Exception in Medium LLM call: {e}-----")

        # 5) Format any tables in the answer
        answer_body = extract_and_format_table(answer_body)

        # 6) Prefix the final answer with a note about which documents were used
        sources_used = ", ".join(f"\"{src}\"" for src, _ in contexts)
        final_answer = (
            f"📚 จากเอกสารนโยบาย: {sources_used}\n\n"
            f"{answer_body}\n\n"
            f"---\n"
            f"🧠 *DEBUG*\n"
            f"Category: **Medium**\n"
            f"Answer Method: **VectorStore + LLM (top {TOP_K} chunks)**\n"
            f"Vector Score: {top_score:.2f}\n"
            f"Fuzzy Score: {fuzzy_score:.2f}"
        )

        msg = cl.Message(
            content="",
            metadata={"difficulty": "Medium"}
        )
        await msg.send()

        for char in final_answer:
            await msg.stream_token(char)
            await asyncio.sleep(0.005)  # Adjust speed for animation feel

        await msg.update()
        save_conversation_log(thread_id, message.id, role="bot", content=final_answer, difficulty="Medium")
        return

    if level == "Hard":
        # build and send your “hard”‐level answer however you like
        # here’s a simple example that just acknowledges Hard:
        final_answer = (
            f"✅ Here’s a deep‐dive answer (Hard level): …\n\n"
            f"🧠 *DEBUG*\n"
            f"Category: **Hard**\n"
            f"Vector Score: {top_score:.2f}\n"
            f"Fuzzy Score: {fuzzy_score:.2f}"
        )
        msg = cl.Message(
            content="",
            metadata={"difficulty": "Hard"}
        )
        await msg.send()

        for char in final_answer:
            await msg.stream_token(char)
            await asyncio.sleep(0.005)  # Optional: adjust for animation speed

        await msg.update()

        save_conversation_log(
            thread_id,
            message.id,
            role="bot",
            content=final_answer,
            difficulty="Hard"
        )
        return


shown_admin_replies = {}

async def poll_admin_reply(thread_id: str, parent_id: str):
    key = f"admin-reply:{thread_id}:{parent_id}"
    print(f"🧠 Start polling Redis for admin reply on key: {key}")

    for attempt in range(60):  # up to 5 minutes
        try:
            raw = redis_client.get(key)
            if raw:
                payload = json.loads(raw.decode("utf-8"))
                parent_content = payload.get("parent_content")
                replies = payload.get("replies", [])

                if not replies:
                    print("⚠️ No replies found in payload.")
                    await asyncio.sleep(5)
                    continue

                # Optional deduplication: Use latest reply's timestamp or id
                last_id = replies[-1]["id"]
                if shown_admin_replies.get(key) == last_id:
                    print("🔁 Already shown latest reply. Waiting for update...")
                    await asyncio.sleep(5)
                    continue

                # ⬆️ Push the original parent message
                await send_with_feedback(
                    content=f"🧾 Original Question:\n\n{parent_content}",
                    author="User"
                )
                save_conversation_log(thread_id, parent_id, role="admin", content=cleaned)

                # ⬇️ Push each reply (sorted by timestamp if needed)
                for r in replies:
                    content = r.get("body", {}).get("content", "")
                    if content:
                        await send_with_feedback(
                            content=f"📬 Reply from Admin:\n\n{content}",
                            author="Admin",
                            parent_id=parent_id
                        )

                shown_admin_replies[key] = last_id

                # ✅ Save admin conversation log in Redis
                conv_log = {
                    "thread_id": thread_id,
                    "parent_id": parent_id,
                    "user_question": parent_content,
                    "admin_replies": replies,
                    "timestamp": time.time()
                }
                redis_client.setex(f"log:admin_reply:{thread_id}:{parent_id}", 86400, json.dumps(conv_log))
                print("📥 Saved admin conversation log.")

                return

            else:
                print(f"⏳ ({attempt + 1}/60) No reply yet for key: {key}")

        except Exception as e:
            print(f"❌ Error polling admin reply: {e}")

        await asyncio.sleep(5)

    print(f"⌛ Timeout: No admin reply found after 60 attempts for key: {key}")


