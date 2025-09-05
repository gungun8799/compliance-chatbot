# Run application locally using this command: chainlit run app.py -h --root-path /compliance/chat
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
from collections import defaultdict


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

# OneLogin OAuth Authentication
from auth.onelogin_oauth_provider import OneLoginOAuthProvider
from auth.inject_custom_auth import add_custom_oauth_provider

patch.apply_patch()

# OneLogin OAuth provider will be initialized after logger is set up

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

# Initialize OneLogin OAuth provider (with graceful fallback)
try:
    add_custom_oauth_provider("onelogin", OneLoginOAuthProvider())
    logger.info("✅ OneLogin OAuth provider initialized successfully")
except ValueError as e:
    logger.warning(f"⚠️ OneLogin OAuth not configured: {e}")
    logger.warning("⚠️ Application will run without OAuth authentication")
except Exception as e:
    logger.error(f"❌ Error initializing OneLogin OAuth provider: {e}")
    logger.warning("⚠️ Application will run without OAuth authentication")

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
TOKEN_LIMIT = 4096 # Increased token limit for better memory retention
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
VECTOR_MIN_THRESHOLD = 0.4
VECTOR_MEDIUM_THRESHOLD = 0.7
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
        "FA-G-12 Entertainment Policy_TH_15012025_for chatbot.docx",
        "FA-G-10 เงินทดรองจ่ายพนักงาน (Chat bot).docx",
        "FA-G-01 Petty Cash (Chat bot).docx",
        "การเบิกค่าใช้จ่ายพนักงาน.docx",
        "FAQ_Narrative_1 DoA LOA_Proj App_Payment App.docx",
        "FAQ_Narrative - Org Structure and General Question.docx",
        "FA-G-11 การเบิกจ่ายโดยไม่มีใบสั่งซื้อ (Chat bot).docx",
        "management chatbot.docx"
    ],
    "คู่ค้าซื้อมาขายไป (Commercial / Trade Supplier)": [
        "การเพิ่มข้อมูลคู่ค้า และการจ่ายเงิน (Trade).docx",
        "FAQ_Narrative_2 Trade Supplier.docx",
        "FAQ_Narrative - Org Structure and General Question.docx",
        "management chatbot.docx"
    ],
    "คู่ค้าอื่นๆ (Procurement / Non-Trade Supplier)": [
        "การเพิ่มและแก้ไขข้อมูลคู่ค้า (Non-trade).docx",
        "FAQ_Narrative_3 Non Trade Supplier.docx",
        "FAQ_Narrative - Org Structure and General Question.docx",
        "management chatbot.docx"
        
    ],
    "ลูกค้าผู้เช่าพื้นที่ (Mall / Tenant)": [
        "การเพิ่ม คัดเลือกลูกค้า การต่อสัญญา และการติดตามหนี้.docx",
        "FAQ_Narrative_4 Mall.docx",
        "FA-G-20_Write-off Tenant deposit_TH_15022025_for chatbot.docx",
        "FAQ_Narrative - Org Structure and General Question.docx",
        "management chatbot.docx"
        
    ],
    "ลูกค้า B2B": [
        "การบริหารสินเชื่อสำหรับธุรกิจ B2B.docx",
        "PFAQ_Narrative - Org Structure and General Question.docx",
        "B2B Others.docx",
        "management chatbot.docx"
    ],
    "ลูกหนี้อื่นๆ (AR Others / AR non-mall)": [
        "การเพิ่มและแก้ไขข้อมูลคู่ค้า (Non-trade).docx",
        "FAQ_Narrative - Org Structure and General Question.docx",
        "FA-G-14_Central Debtor Code for Invoice Issuance (TH)_01022025_for chatbot.docx",
        "management chatbot.docx"
    ],
    "สินทรัพย์ (Asset)": [
        "FA-G-13 - Asset management policy.docx",
        "FAQ_Narrative - Org Structure and General Question.docx",
        "FAQ_Narrative_7 Asset.docx",
        "management chatbot.docx"
    ]
}

# Redis Client
redis_client = redis.Redis.from_url(
    REDIS_CHATSTORE_URI,
    decode_responses=True  # Optional: returns strings instead of bytes
)

"""
#parsed_redis_url = urlparse(REDIS_CHATSTORE_URI)
redis_client = redis.Redis(
    host=parsed_redis_url.hostname,
    port=parsed_redis_url.port or 6379,
    password=REDIS_CHATSTORE_PASSWORD,
    db=0,
)
"""

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

# File-based memory storage as fallback
import tempfile
import pickle
from pathlib import Path

MEMORY_DIR = Path(tempfile.gettempdir()) / "chainlit_memory"
MEMORY_DIR.mkdir(exist_ok=True)

def save_memory_to_file(memory_key: str, messages: list):
    """Save chat messages to file"""
    try:
        memory_file = MEMORY_DIR / f"{memory_key}.pkl"
        with open(memory_file, 'wb') as f:
            pickle.dump(messages, f)
        logger.info(f"💾 Saved {len(messages)} messages to file: {memory_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save memory to file: {e}")

def load_memory_from_file(memory_key: str) -> list:
    """Load chat messages from file"""
    try:
        memory_file = MEMORY_DIR / f"{memory_key}.pkl"
        if memory_file.exists():
            with open(memory_file, 'rb') as f:
                messages = pickle.load(f)
            logger.info(f"📂 Loaded {len(messages)} messages from file: {memory_file}")
            return messages
        else:
            logger.info(f"📂 No memory file found: {memory_file}")
            return []
    except Exception as e:
        logger.error(f"❌ Failed to load memory from file: {e}")
        return []

# Chat Store - Test Redis connectivity with file fallback
try:
    logger.info(f"🔌 Testing Redis connection to: {REDIS_CHATSTORE_URI}")
    chat_store = RedisChatStore(
        redis_url=REDIS_CHATSTORE_URI, db=0, password=REDIS_CHATSTORE_PASSWORD, ttl=180
    )
    # Test the connection
    test_key = "test_connection"
    redis_client.set(test_key, "test_value", ex=5)  # 5 second expiry
    test_result = redis_client.get(test_key)
    if test_result:
        logger.info("✅ Redis connection successful!")
        redis_client.delete(test_key)
        use_file_fallback = False
    else:
        logger.warning("⚠️ Redis test failed - using file-based memory")
        chat_store = None
        use_file_fallback = True
except Exception as e:
    logger.error(f"❌ Redis connection failed: {e}")
    logger.info("🔄 Chat memory will use file-based fallback")
    chat_store = None
    use_file_fallback = True

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
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    },
    "Accounting Compliance": {
        "context_prompt": SYSTEM_PROMPT_STANDARD,
        "welcome_message": "Hi there! Need help with accounting compliance?",
        "llm_settings": {
            "model": "default",
            "api_base": "https://api-cpxis.lotuss.com/llm/v1",
            "api_key": "accounting.lotuss.F51DAF28FD6422DDF3CD864F833CC",
            "is_chat_model": True,
            "is_function_calling_model": False,
            "temperature": 0.2,
            "http_client": httpx.Client(verify=False),
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    },
}

async def send_animated_message(
    base_msg: str,
    frames: list,
    interval: float = 0.8
) -> None:
    """Displays an animated message optimized for performance."""
    logger.info(f"🎬 ANIMATION FUNCTION CALLED with msg='{base_msg}', frames={frames}")
    
    msg = cl.Message(content=base_msg, author="Customer Service Agent")
    await msg.send()
    logger.info(f"🎬 ANIMATION MESSAGE SENT: {msg.id}")

    progress = 0
    bar_length = 12

    try:
        iteration = 0
        while True:
            iteration += 1
            current_frame = frames[progress % len(frames)]
            progress_bar = ("▣" * (progress % bar_length)).ljust(bar_length, "▢")
            # Update the content property, then issue a plain update()
            new_content = f"{current_frame} {base_msg}\n{progress_bar}"
            logger.info(f"🎬 ANIMATION UPDATE {iteration}: '{new_content[:50]}...'")
            msg.content = new_content
            await msg.update()
            progress += 1
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info(f"🎬 ANIMATION CANCELLED after {iteration} iterations")
        # Final static display when the task is cancelled
        msg.content = base_msg
        await msg.update()

async def stream_text_response(message: cl.Message, text: str, delay: float = 0.05, mode: str = "word"):
    """Stream text to a message with typing effect."""
    import asyncio
    
    # Start with empty content
    message.content = ""
    await message.update()
    
    if mode == "word":
        # Stream word by word while preserving line breaks
        import re
        # Split by spaces but preserve line breaks
        parts = re.split(r'(\s+)', text)  # This keeps the separators (spaces/newlines)
        current_text = ""
        for i, part in enumerate(parts):
            current_text += part
            message.content = current_text
            await message.update()
            if i < len(parts) - 1 and part.strip():  # Don't delay on whitespace or last part
                await asyncio.sleep(delay)
    else:
        # Stream character by character (for shorter text)
        for i in range(len(text) + 1):
            message.content = text[:i]
            await message.update()
            if i < len(text):  # Don't delay after the last character
                await asyncio.sleep(delay)

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
    # Display BU selection prompt without animation
    full_prompt = "กรุณาเลือกหัวข้อคำถามโดยพิมพ์ตัวเลขเพื่อเลือกหัวข้อ:\n\n" + options
    bu_message = cl.Message(content=full_prompt)
    await bu_message.send()

async def handle_bu_selection(text: str):
    """Handle business unit selection from user input."""
    business_units = cl.user_session.get("business_units", [])
    
    try:
        selection = int(text) - 1
        if 0 <= selection < len(business_units):
            selected_bu = business_units[selection]
            cl.user_session.set("selected_bu", selected_bu)
            cl.user_session.set("awaiting_bu_selection", False)
            
            logger.info(f"✅ Selected BU: {selected_bu}")
            
            # Welcome message after BU selection
            welcome_msg = f"""✅ เลือก **{selected_bu}** เรียบร้อยแล้ว

ตอนนี้คุณสามารถถามคำถามเกี่ยวกับนโยบายและระเบียบปฏิบัติได้เลย ผมจะช่วยตอบและหากต้องการข้อมูลเพิ่มเติมเพื่อให้คำตอบที่ตรงประเด็น ผมจะถามกลับครับ

กรุณาพิมพ์คำถามของคุณ 👇"""
            
            # Display welcome message without animation
            welcome_message = cl.Message(content=welcome_msg)
            await welcome_message.send()
        else:
            error_msg = cl.Message(content="❌ กรุณาเลือกหมายเลข 1-7 เท่านั้น")
            await error_msg.send()
            await ask_business_unit()
            
    except ValueError:
        error_msg = cl.Message(content="")
        await error_msg.send()
        await stream_text_response(error_msg, "❌ กรุณาเลือกหมายเลข 1-7 เท่านั้น")
        await ask_business_unit()

async def handle_policy_question(text: str, selected_bu: str):
    """Handle policy questions with natural conversation flow."""
    logger.info(f"🔍 Processing policy question: {text} for BU: {selected_bu}")
    
    try:
        # Start animated message as a background task (using the working approach)
        import contextlib
        logger.info("🎬 Starting document retrieval animation...")
        animation_task = asyncio.create_task(
            send_animated_message(
                base_msg="กำลังเช็ค Policy ให้อยู่ รอสักครู่นะครับ …",
                frames=["🔍", "🔎", "📄", "📋", "📊"],
                interval=0.8  # Slower interval to see the animation
            )
        )
        
        try:
            # Get top 10 relevant nodes from selected BU documents  
            import time
            start_time = time.time()
            logger.info("⏰ Starting document retrieval...")
            
            # Add minimum delay to ensure animation shows
            await asyncio.sleep(2.0)  # Minimum 2 seconds to see animation
            
            nodes = await retrieve_bu_filtered_nodes(text, selected_bu, top_k=10)
            end_time = time.time()
            logger.info(f"⏰ Document retrieval took {end_time - start_time:.2f} seconds")
        finally:
            # Stop animation using the working pattern
            logger.info("🛑 Stopping document retrieval animation...")
            animation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await animation_task
        
        if not nodes:
            # Create new message for no results
            no_results_msg = cl.Message(content="", author="Customer Service Agent")
            await no_results_msg.send()
            no_info_msg = f"ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถาม '{text}' ในหมวด {selected_bu} กรุณาลองถามในรูปแบบอื่นหรือติดต่อทีมนโยบายโดยตรง"
            await stream_text_response(no_results_msg, no_info_msg)
            return
            
        # Get chat memory for conversation history  
        memory = cl.user_session.get("memory")
        conversation_history = ""
        if memory:
            try:
                messages = memory.get()
                logger.info(f"🧠 Total messages in memory: {len(messages)}")
                if messages:
                    # Get last 8 messages for context
                    recent_messages = messages[-8:] 
                    # Format conversation history more clearly
                    formatted_messages = []
                    for i, msg in enumerate(recent_messages):
                        role = "User" if msg.role == "user" else "Assistant"
                        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                        formatted_messages.append(f"{role}: {content}")
                        logger.info(f"🧠 Message {i+1}: {role}: {msg.content[:50]}...")
                    conversation_history = "\n".join(formatted_messages)
                    logger.info(f"🧠 Conversation history being passed to LLM ({len(formatted_messages)} messages)")
                else:
                    logger.info("🧠 Memory exists but no messages stored yet")
            except Exception as e:
                logger.error(f"🧠 Error retrieving memory: {e}")
                conversation_history = ""
        else:
            logger.info("🧠 No memory found - empty conversation history")
        
        # Create the natural policy expert prompt with nodes and question
        policy_prompt = create_new_policy_prompt(text, nodes, conversation_history, selected_bu)
        logger.info(f"🤖 Full prompt being sent to LLM: {policy_prompt[:500]}...")  # First 500 chars
        
        # Start animated message for LLM processing
        logger.info("🎬 Starting LLM processing animation...")
        llm_animation_task = asyncio.create_task(
            send_animated_message(
                base_msg="กำลังประมวลผลและเตรียมคำตอบ...",
                frames=["🤖", "💭", "✨", "⚡", "🧠"],
                interval=0.8  # Slower interval to see the animation
            )
        )
        
        try:
            # Get LLM response
            start_llm_time = time.time()
            logger.info("⏰ Starting LLM processing...")
            
            # Add minimum delay to ensure animation shows 
            await asyncio.sleep(2.0)  # Minimum 2 seconds to see animation
            
            llm = get_llm_settings("Accounting Compliance")
            response = llm.chat([ChatMessage(role="user", content=policy_prompt)])
            answer = response.message.content.strip()
            
            end_llm_time = time.time()
            logger.info(f"⏰ LLM processing took {end_llm_time - start_llm_time:.2f} seconds")
        finally:
            # Stop LLM animation
            logger.info("🛑 Stopping LLM processing animation...")
            llm_animation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await llm_animation_task
        
        # Create new message for streaming the response
        response_msg = cl.Message(content="", author="Customer Service Agent")
        await response_msg.send()
        
        # Stream the answer with typing effect
        await stream_text_response(response_msg, answer)
        
        # Add assistant response to memory
        memory = cl.user_session.get("memory")
        if memory:
            try:
                memory.put(ChatMessage(role="assistant", content=answer))
                # Verify the message was stored
                messages_after = memory.get()
                logger.info(f"✅ Added assistant response to memory. Total messages now: {len(messages_after)}")
                if messages_after:
                    logger.info(f"🧠 Latest message: {messages_after[-1].role}: {messages_after[-1].content[:100]}...")
            except Exception as e:
                logger.error(f"❌ Failed to add assistant response to memory: {e}")
        else:
            logger.error("❌ No memory found when trying to save assistant response")
        # Log the conversation
        thread_id = cl.context.session.thread_id
        save_conversation_log(thread_id, response_msg.id, role="assistant", content=answer)
        
        logger.info(f"✅ Successfully answered policy question")
        
    except Exception as e:
        logger.error(f"❌ Error handling policy question: {e}")
        error_msg = "ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำถาม กรุณาลองใหม่อีกครั้ง"
        # Cancel any running animations
        if 'animation_task' in locals():
            animation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await animation_task
        if 'llm_animation_task' in locals():
            llm_animation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await llm_animation_task
        
        if 'response_msg' in locals():
            response_msg.content = error_msg
            await response_msg.update()
        else:
            await cl.Message(content=error_msg).send()

async def retrieve_bu_filtered_nodes(query: str, selected_bu: str, top_k: int = 5):
    """Retrieve top nodes filtered by business unit documents."""
    try:
        # Get allowed documents for the selected BU
        allowed_docs = BU_DOCUMENT_MAP.get(selected_bu, [])
        if not allowed_docs:
            logger.warning(f"No documents found for BU: {selected_bu}")
            return []
            
        logger.info(f"📁 Allowed documents for '{selected_bu}': {allowed_docs}")
        
        # Get vector store and create index
        dataset = DATASET_MAPPING.get("Accounting Compliance", QDRANT_COLLECTION_NAME)
        vector_store = qdrant_manager.get_vector_store(dataset, hybrid=True)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Create retriever and get nodes
        retriever = index.as_retriever(similarity_top_k=top_k * 3)  # Get more to filter
        nodes = retriever.retrieve(query)
        
        # Filter nodes by BU documents
        filtered_nodes = [n for n in nodes if n.node.metadata.get("source") in allowed_docs]
        
        # Take top k after filtering
        filtered_nodes = filtered_nodes[:top_k]
        
        logger.info(f"🔍 Retrieved {len(filtered_nodes)} filtered nodes for query: {query}")
        
        # Log the retrieved nodes
        for i, node in enumerate(filtered_nodes):
            source = node.node.metadata.get("source", "unknown")
            score = getattr(node, 'score', 0.0)
            logger.info(f"  Node {i+1}: {source} (score: {score:.3f})")
        
        return filtered_nodes
        
    except Exception as e:
        logger.error(f"❌ Error retrieving nodes: {e}")
        return []

def create_context_from_nodes(nodes):
    """Create formatted context from retrieved nodes."""
    if not nodes:
        return "ไม่พบข้อมูลที่เกี่ยวข้อง"
    
    context_parts = []
    for i, node in enumerate(nodes, 1):
        source = node.node.metadata.get("source", "unknown")
        section_path = node.node.metadata.get("section_path", [])
        section_str = " > ".join(section_path) if section_path else "ไม่ระบุหัวข้อ"
        
        context_parts.append(f"""--- เอกสารที่ {i} ---
แหล่งข้อมูล: {source}
หัวข้อ: {section_str}
เนื้อหา: {node.node.text}""")
    
    return "\n\n".join(context_parts)

def create_policy_expert_prompt(user_question: str, context: str, conversation_history: str, selected_bu: str):
    """Create a natural policy expert prompt for the LLM."""
    
    prompt = f"""คุณคือผู้ช่วยด้านนโยบายบริษัทที่เป็นมิตร สำหรับหมวด "{selected_bu}"

🎯 **สำคัญมาก: ตอบแบบสนทนาธรรมชาติเหมือนคนจริงๆ ไม่ใช่เอกสารหรือคู่มือ**

วิธีการตอบของคุณ:
• พูดแบบเพื่อนร่วมงานที่เป็นกันเอง ไม่เป็นทางการ
• อธิบายแบบง่ายๆ ไม่ต้องมีโครงสร้างซับซ้อน  
• เมื่อไม่แน่ใจ ถามกลับแบบเป็นมิตร

💬 ประวัติการสนทนา:
{conversation_history}

❓ คำถาม: {user_question}

📋 ข้อมูลจากเอกสารนโยบาย:
{context}

💡 ตัวอย่างการตอบที่ดี:
"ครับ โครงการ 300 ล้านบาทนี้ต้องให้ CP Axtra ExCom เป็นผู้อนุมัติครับ เพราะเป็นโครงการขนาดใหญ่"
"เรื่องนี้มีหลายกรณีเลย ช่วยบอกเพิ่มเติมหน่อยได้ไหมครับว่าโครงการประเภทไหน?"
"ขออภัยครับ ไม่เห็นข้อมูลนี้ในเอกสาร ลองติดต่อทีมนโยบายโดยตรงดูครับ"

🚫 **ห้ามใช้รูปแบบเหล่านี้เด็ดขาด:**
- ห้าม: "### 📌 รายละเอียด", "### 💡 ขั้นตอน", "### ❓ ถ้ายัง"  
- ห้าม: "1. 2. 3." หรือ "- ช่วงงบประมาณ:" แบบทางการ
- ห้าม: การแสดงรายการแบบเทคนิคหรือดัมพ์ข้อมูลดิบ

✅ **ใช้แบบนี้แทน:**
ตัวอย่างการตอบที่ถูกต้อง:
"ครับ โครงการ 300 ล้านบาทนี้ต้องให้ CP Axtra ExCom เป็นผู้อนุมัติครับ

เหตุผลก็คือเพราะโครงการมูลค่าในช่วงนี้ถือว่าเป็นโครงการขนาดใหญ่ จึงต้องให้คณะกรรมการระดับสูงเป็นผู้ตัดสินใจครับ

แต่ถ้าอยากให้คำตอบที่แม่นยำกว่านี้ ช่วยบอกหน่อยได้ไหมครับว่าโครงการนี้เป็นประเภทไหน เช่น สร้างสโตร์ใหม่ หรือโครงการเทคโนโลยี?"

**ตอบแบบสนทนาธรรมชาติเหมือนคนจริงๆ ไม่ใช่แบบเอกสารหรือคู่มือ**

**CRITICAL: Write your response as if you're having a friendly conversation with a colleague. NO bullet points, NO numbered lists, NO markdown headers (###). Just natural Thai conversation.**

กรุณาตอบ:"""

    return prompt

def create_policy_expert_prompt_with_nodes(user_question: str, nodes: list, conversation_history: str, selected_bu: str):
    """Create prompt that sends nodes directly to LLM for better processing."""
    
    # Format nodes for LLM
    nodes_data = ""
    for i, node in enumerate(nodes, 1):
        source = node.node.metadata.get("source", "unknown")
        section_path = node.node.metadata.get("section_path", [])
        section_str = " > ".join(section_path) if section_path else "ไม่ระบุหัวข้อ"
        score = getattr(node, 'score', 0.0)
        
        nodes_data += f"""
เอกสารที่ {i}:
ไฟล์: {source}
หัวข้อ: {section_str}
ความเกี่ยวข้อง: {score:.3f}
เนื้อหา: {node.node.text}

---"""

    prompt = f"""คุณเป็นผู้ช่วยด้านนโยบาย "{selected_bu}" ที่เป็นมิตรและเชี่ยวชาญ

🎯 ภารกิจ: ตอบคำถามแบบสนทนาธรรมชาติ ไม่ใช่แบบเอกสาร

ประวัติการสนทนา: {conversation_history}

คำถามของผู้ใช้: {user_question}

ข้อมูลจากเอกสาร: {nodes_data}

🚨 สำคัญมาก - รูปแบบการตอบ:
• ตอบแบบเพื่อนคุยกันเป็นกันเอง
• ไม่ใช้ "### หัวข้อ" หรือ bullet points ซับซ้อน  
• ถ้าข้อมูลไม่ชัดเจนหรือมีหลายกรณี → ถามกลับแบบเป็นมิتร
• ถ้ามีข้อมูลชัดเจน → ตอบตรงประเด็นแล้วถามว่าต้องการทราบอะไรเพิ่ม

ตัวอย่างการตอบที่ดี:
"ครับ โครงการ 300 ล้านนี้ต้องให้ CP Axtra ExCom อนุมัติครับ เพราะอยู่ในช่วงโครงการขนาดใหญ่ 

แต่จริงๆ แล้วขึ้นอยู่กับว่าเป็นโครงการประเภทไหนด้วย เช่น ถ้าเป็นโครงการสร้างสโตร์หรือโครงการเทคโนโลยี กฎอาจแตกต่างกันหน่อย

ช่วยบอกได้ไหมครับว่าโครงการนี้เป็นแบบไหน?"

กรุณาตอบ:"""

    return prompt

def create_new_policy_prompt(user_question: str, nodes: list, conversation_history: str, selected_bu: str):
    """Create the definitive prompt for natural policy responses."""
    
    # Format nodes for LLM
    nodes_data = ""
    for i, node in enumerate(nodes, 1):
        source = node.node.metadata.get("source", "unknown")
        section_path = node.node.metadata.get("section_path", [])
        section_str = " > ".join(section_path) if section_path else "No section"
        score = getattr(node, 'score', 0.0)
        
        nodes_data += f"""
Document {i}:
Source: {source}
Section: {section_str}
Relevance: {score:.3f}
Content: {node.node.text}

---"""

    prompt = f"""You are a friendly company policy assistant for "{selected_bu}". 😊

CONVERSATION HISTORY: {conversation_history}

USER QUESTION: {user_question}

RETRIEVED DOCUMENTS: {nodes_data}

INSTRUCTIONS:

1. LANGUAGE: Respond in the same language as the user's question. Thai question = Thai response.

2. STYLE: Natural, friendly conversation like a helpful colleague. Use appropriate emojis to make responses more engaging! 😊

3. RESPONSE STRATEGY & INTELLIGENT CLARIFICATION: 
   
   **STEP 1: ANALYZE THE RETRIEVED DOCUMENTS**
   Before responding, carefully examine the retrieved documents to identify:
   - Multiple possible scenarios/categories mentioned
   - Different approval levels or amount ranges
   - Various document types or procedures
   - Multiple stakeholders or departments involved
   - Conditional requirements (if X then Y)
   
   **STEP 2: DETERMINE RESPONSE TYPE**
   - GENERAL/BROAD questions (e.g., "Procurement ดูอะไรบ้าง") = Give HIGH-LEVEL summary + ask for specifics
   - AMBIGUOUS questions where documents show MULTIPLE OPTIONS = Ask clarification questions
   - SPECIFIC questions with clear context = Provide detailed answers
   - Follow-up questions = Connect to conversation history
   
   **STEP 3: INTELLIGENT CLARIFICATION DETECTION**
   ASK CLARIFICATION when you find in the documents:
   - Multiple approval authorities for different amount ranges
   - Different procedures for different types of projects/suppliers/customers
   - Conditional requirements that depend on specific criteria
   - Multiple document requirements for different scenarios
   
   **CLARIFICATION EXAMPLES:**
   ❌ Don't give generic answers when documents show multiple options
   ✅ Ask: "เนื่องจากมีขั้นตอนที่แตกต่างกันตามประเภท ช่วยบอกหน่อยได้ไหมครับว่า..."
   
   **WHEN TO ASK VS WHEN TO ANSWER:**
   - If documents mention "depends on", "varies by", "different types" → ASK FOR CLARIFICATION
   - If user provides specific amount, type, or scenario → GIVE DETAILED ANSWER
   - If documents show clear single procedure → GIVE DIRECT ANSWER

4. CRITICAL FORMATTING REQUIREMENTS: 
   - MANDATORY: Each bullet point MUST be on a separate line with line break
   - MANDATORY: Use double line breaks (\n\n) between sections
   
   BULLET POINT FORMAT - FOLLOW EXACTLY:
   ✅ CORRECT FORMAT:
   "เอกสารที่ต้องตรวจสอบ:
   
   • สัญญา (Contract) - เพื่อยืนยันเงื่อนไขการทำงาน
   • Confidentiality Agreement (NDA) - เพื่อความปลอดภัย
   • D&B Report - รายงานเครดิต
   
   ข้อมูลเพิ่มเติม:"
   
   ❌ ABSOLUTELY FORBIDDEN:
   "• Item 1 • Item 2 • Item 3" (all on one line)
   
   - For tables, use proper markdown table format with line breaks
   - Use **bold text** for headings and emphasis  
   - Add blank lines between paragraphs
   - NO "### headers", NO "1.2.3." numbered lists

5. INTELLIGENT CLARIFICATION EXAMPLES:

**SCENARIO 1: User asks "โครงการ 300 ล้านใครอนุมัติ"**
IF documents show different authorities for different project types:
✅ GOOD - Ask for clarification:
"ครับ ยินดีช่วยเรื่องโครงการ 300 ล้านบาทครับ! 😊

เนื่องจากผู้อนุมัติจะแตกต่างกันตามประเภทโครงการ ช่วยบอกหน่อยได้ไหมครับว่าเป็นโครงการประเภทไหน? 🤔

**จากข้อมูลที่พบ มีประเภทหลักๆ เช่น:**

• โครงการสร้างสโตร์ใหม่
• โครงการเทคโนโลยี/IT  
• โครงการปรับปรุงอุปกรณ์
• โครงการการตลาด

เมื่อทราบประเภทแล้ว จะบอกผู้อนุมัติที่ถูกต้องให้เลยครับ"

**SCENARIO 2: User asks "เพิ่มคู่ค้าใหม่ต้องทำยังไง"**
IF documents show different processes for Trade vs Non-Trade:
✅ GOOD - Ask for clarification:
"ครับ ยินดีช่วยเรื่องการเพิ่มคู่ค้าใหม่ครับ! 😊

ขั้นตอนจะแตกต่างกันตามประเภทคู่ค้า ช่วยบอกหน่อยได้ไหมครับว่าเป็นคู่ค้าประเภทไหน? 🤔

**ประเภทคู่ค้าที่มี:**

• Trade Supplier (คู่ค้าซื้อมาขายไป) 📦
• Non-Trade Supplier (คู่ค้าบริการ/อุปกรณ์) 🔧
• Overseas Supplier (คู่ค้าต่างประเทศ) 🌏

เอกสารและขั้นตอนจะแตกต่างกันนะครับ 📋"

**SCENARIO 3: User asks "อนุมัติค่าใช้จ่าย 50,000 บาท"**
IF documents show different authorities for different expense types:
✅ GOOD - Ask for clarification:
"ครับ สำหรับการอนุมัติ 50,000 บาท 💰

ผู้อนุมัติจะขึ้นอยู่กับประเภทค่าใช้จ่าย ช่วยบอกหน่อยได้ไหมครับว่าเป็น: 🤔

**ประเภทค่าใช้จ่าย:**

• ค่าใช้จ่ายทั่วไป (General Expense) 📝
• Purchase Requisition (การสั่งซื้อ) 🛒
• ค่าใช้จ่ายพนักงาน (Employee Expense) 👤
• ค่าลงทุนโครงการ (Project Investment) 🏗️

แต่ละประเภทมีผู้อนุมัติต่างกันนะครับ ✅"

GOOD - Direct answer when clear:
"**สำหรับโครงการ IT มูลค่า 300 ล้านบาท** ต้องได้รับอนุมัติจาก IT&DC Committee ครับ 💻

ใช้เวลาประมาณ 1-2 สัปดาห์ ขึ้นอยู่กับความครบถ้วนของเอกสาร ⏱️

มีอะไรเพิ่มเติมเกี่ยวกับกระบวนการที่อยากทราบไหมครับ? 😊"

GOOD - Follow-up response (connects to previous context):
"เข้าใจแล้วครับ! 👍 **สำหรับโครงการสร้างสโตร์ใหม่ มูลค่า 300 ล้านบาท** 🏪 

[Read the documents to find the exact approval authority for this amount and project type]

[Explain the reasoning based on the policy ranges found in the documents]

[Provide any additional relevant process information from the documents]

มีอะไรเพิ่มเติมที่อยากทราบไหมครับ?"

6. CONVERSATION MEMORY & CONTEXT:
   - ALWAYS refer back to previous conversation when answering follow-up questions
   - If user refers to "it", "that", "this", "นั้น", "นี้", "อันนี้" - connect to the previous topic
   - Maintain context throughout the conversation
   - Example: If user previously asked about "ซื้อเครื่องพิมพ์" and now asks "เอกสารประกอบมีอะไรบ้าง", you MUST understand they're asking about documents for buying a printer
   - Check conversation history to understand the full context before responding

CRITICAL DOCUMENT ANALYSIS & PROCESSING:

**BEFORE RESPONDING - ANALYZE DOCUMENTS FOR:**
1. **Multiple Categories/Types**: Look for phrases like "แบ่งเป็น", "ประเภท", "กรณี", "depends on", "varies"
2. **Amount Ranges**: Different procedures for different monetary values
3. **Conditional Logic**: "ถ้า...แล้ว", "หาก", "เมื่อ", "กรณีที่"
4. **Different Stakeholders**: Multiple departments, committees, approval levels
5. **Geographic/Location Differences**: Domestic vs International, different regions

**DECISION TREE:**
- IF documents show ONE clear answer → Provide direct answer
- IF documents show MULTIPLE procedures/authorities → ASK CLARIFICATION with specific options from documents
- IF user question is vague AND documents have multiple scenarios → ASK CLARIFICATION
- IF user provides specifics that match document conditions → Give targeted answer

**FORMATTING REQUIREMENTS:**
- ALWAYS preserve line breaks when listing items or bullet points
- Each bullet point must be on its own line  
- Use blank lines to separate sections for readability
- Use **bold text** for categories and emphasis
- When asking clarification, always list the actual options found in documents

**PROCESSING STEPS:**
- STEP 1: Scan all documents for categories, conditions, and variations
- STEP 2: Check if user question maps to specific scenario or is ambiguous
- STEP 3: If ambiguous + multiple options exist → ask clarification with document-based options
- STEP 4: If specific → find exact match and provide detailed answer
- STEP 5: Always explain reasoning based on document evidence
- NEVER guess or assume - only use information explicitly stated in the documents

For follow-up questions, always connect to previous conversation context. For new ambiguous questions, ask for clarification first.

CRITICAL INTELLIGENCE REQUIREMENTS: 

**MANDATORY CLARIFICATION TRIGGERS:**
- User asks about "โครงการ" but doesn't specify type → ASK what type of project
- User asks about "คู่ค้า" but doesn't specify Trade/Non-Trade → ASK which type
- User asks about "อนุมัติ" with amount but no expense type → ASK expense category  
- User asks about "เอกสาร" but scenario unclear → ASK for specific situation
- Documents show different processes for different criteria → ALWAYS ASK FOR CLARIFICATION

**RESPONSE PATTERNS:**
- BROAD questions ("ดูอะไรบ้าง", "อนุมัติอะไรบ้าง") = High-level summary + ask for specifics
- AMBIGUOUS questions with multiple document scenarios = Ask clarification with document-based options
- SPECIFIC questions with clear context = Detailed answer with document evidence
- Always explain WHY you're asking for clarification (because different types have different requirements)

**INTELLIGENT BEHAVIOR:**
- Read documents FIRST to identify variations before responding
- If you find multiple procedures/authorities → Don't guess, ASK for clarification
- Base clarification options on ACTUAL categories found in retrieved documents
- Connect follow-ups to conversation history for context

YOUR RESPONSE:"""

    return prompt
        
# ✅ Add this for on-demand manual retrieval testin
def manual_retrieve(query: str, top_k=5):
    from llama_index.core import Settings, VectorStoreIndex
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
    # Clear known clarification-related session keys
    for key in [
        "awaiting_clarification",
        "clarification_rounds",
        "fuzzy_clarification_rounds",
        "possible_summaries",
        "nodes_to_consider",
        "summary_to_meta",
        "original_query",
        "clarification_level",
        "auto_skipped",
        "last_was_clarify",
        "filtered_nodes",
        "hier_sections",
        "pre_drill_nodes",

        "h1_options",
        "drill_level",
        "clarification_just_exited",

    ]:
        cl.user_session.set(key, None)

    # Explicit flags to help downstream logic
    cl.user_session.set("awaiting_clarification", False)
    cl.user_session.set("clarification_just_exited", True)
    cl.user_session.set("clarification_rounds", 0)
    cl.user_session.set("fuzzy_clarification_rounds", 0)
    cl.user_session.set("summary_to_meta", {})
    cl.user_session.set("possible_summaries", [])


def is_valid_user_question(text: Optional[str]) -> bool:
    if not text:
        return False
    stripped = text.strip()
    return (
        len(stripped) > 3
        and not stripped.isdigit()
        and not stripped.lower().startswith("clarified:")
    )

async def answer_from_node(node_or_nodes, user_q: str):
    # In answer_from_node(...)
    # Choose final main question to display
    nodes = node_or_nodes if isinstance(node_or_nodes, list) else [node_or_nodes]
    cl.user_session.set("last_answered_context", nodes)
    clarification_just_exited = cl.user_session.get("clarification_just_exited")
    last_answered_context = cl.user_session.get("last_answered_context")

    if clarification_just_exited and last_answered_context:
        main_question = user_q.strip()
        logger.info(f"📌 Overriding main question after clarification: {main_question}")
        cl.user_session.set("clarification_just_exited", False)
    else:
        main_question = cl.user_session.get("original_user_question") or user_q.strip()
    clear_clarification_state()
    cl.user_session.set("awaiting_clarification", False)

    memory = cl.user_session.get("memory")
    if memory is None:
        logger.warning("⚠️ No memory object found in session.")
        return

    await asyncio.sleep(0.05)  # Optional delay

    # 🧠 Build recent chat history (filtered)
    all_messages = memory.get()
    filtered = [
        m for m in all_messages
        if m.role in {"user", "assistant"}
        and not m.content.strip().isdigit()
        and not m.content.strip().lower().startswith("clarified:")
        and len(m.content.strip()) > 3
    ]
    recent_messages = filtered[-6:]
    chat_history = ""
    for msg in recent_messages:
        role = "👤 ผู้ใช้" if msg.role == "user" else "🤖 ผู้ช่วย"
        chat_history += f"{role}: {msg.content.strip()}\n"

    # ✅ Determine latest user question
    latest_user_q = None
    for m in reversed(memory.get()):
        content = m.content.strip()
        if m.role == "user" and content and not content.lower().startswith("clarified:") and not content.isdigit():
            latest_user_q = content
            break

    logger.info(f"📌 latest_user_q = {latest_user_q}")
    logger.info(f"📌 original_user_question = {cl.user_session.get('original_user_question')}")
    # ✅ Update original_user_question ONLY if valid
    # ✅ Update original_user_question ONLY if it hasn't been set already in this clarification flow
    is_clarifying = cl.user_session.get("clarifying", False)
    orig_q = cl.user_session.get("original_user_question")
    should_override = not is_clarifying and is_valid_user_question(latest_user_q)

    # If this is clearly a fresh question (valid + not clarifying), override original_user_question
    if should_override and latest_user_q != orig_q:
        cl.user_session.set("original_user_question", latest_user_q)
        logger.info(f"✅ Overrode original_user_question with new fresh question: {latest_user_q}")
    else:
        logger.info(f"🚫 Skipped overriding original_user_question: is_clarifying={is_clarifying}, latest_user_q={latest_user_q}, existing={orig_q}")
    # ✅ Prepare for LLM call
    runnable = cl.user_session.get("runnable")
    if runnable is None:
        logger.error("❌ 'runnable' is not set in user session!")
        await cl.Message("เกิดข้อผิดพลาดภายในระบบ ไม่สามารถตั้งค่า LLM ได้").send()
        return

    full_text = "\n\n".join(n.node.text.strip().replace("\n", " ") for n in nodes)
    source = nodes[0].node.metadata.get("source", "Unknown")
    logger.info(f"📄 Answer source: {source}")
    logger.info(f"📦 Combined chunk text length: {len(full_text)} characters")

    # ✅ Track selection path
    full_paths = [n.node.metadata.get("section_path", []) for n in nodes]
    section_titles = [p[-1] for p in full_paths if p]
    section_str = " / ".join(section_titles)
    selection_path = cl.user_session.get("selection_path") or []
    if full_paths:
        deepest_path = max(full_paths, key=len)
        path_str = " / ".join(deepest_path)
        if path_str not in selection_path:
            selection_path.append(path_str)
        cl.user_session.set("selection_path", selection_path)
        logger.info(f"📌 Updated selection path memory: {selection_path}")

    # ─── Get final user message from memory ──────────────────────
    memory = cl.user_session.get("memory")
    prior_messages = memory.get()
    main_question = cl.user_session.get("original_user_question") or next(
        (m.content for m in reversed(prior_messages) if m.role == "user" and not m.content.strip().isdigit()),
        user_q
    )
    # ✅ Prompt to LLM
    prompt = (
        f"📜 ประวัติการสนทนา:\n{chat_history}\n\n"
        f'📌 คำถามหลักจากผู้ใช้: "{main_question}"\n\n'
        f'📄 เอกสารนโยบาย: "{source}"\n\n'
        f'เนื้อหาที่เกี่ยวข้องมีดังนี้:\n"""{full_text}\n"""\n\n'
        "กรุณาตอบโดยอ้างอิงรายละเอียดทั้งหมดจากเนื้อหานี้อย่างครบถ้วนและระบุเงื่อนไขที่เกี่ยวข้องให้ชัดเจน "
        "หากมีกรณีหรือเงื่อนไขพิเศษ โปรดแสดงให้ครบทุกกรณี เช่น \"ถ้า...ให้...\" หรือ \"ในกรณีที่...ต้อง...\" "
        "และอย่าสรุปรวมหลายเงื่อนไขเป็นบรรทัดเดียว "
        "**สำคัญมาก:\\n"
        "1. ใช้เฉพาะข้อมูลในเนื้อหาข้างต้นเท่านั้น ห้ามสร้างคำตอบจากความรู้ภายนอกหรือสมมติขึ้นเอง\\n"
        "2. หากเป็นคำถามเกี่ยวกับจำนวนเงิน ให้เปรียบเทียบตัวเลขอย่างระมัดระวัง เช่น 400,000,000 > 100,000,000\\n"
        "3. ตอบด้วยตำแหน่งที่ระบุในเนื้อหาเท่านั้น เช่น Group CEO, CEO, Senior Director ฯลฯ\\n"
        "4. ห้ามใช้คำตอบที่ไม่ได้ระบุในเนื้อหา เช่น CP Axtra ExCom หรือชื่ออื่นๆ ที่ไม่มีในเอกสาร**"
    )
    logger.info(f"🧠 Final LLM prompt = \n{prompt}")

    # ✅ Animation
    frames = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
    animation_task = asyncio.create_task(
        send_animated_message("กำลังเช็ค Policy ให้อยู่ รอสักครู่นะคะ …", frames, interval=0.3)
    )

    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, runnable.query, prompt)
        answer = resp.response if hasattr(resp, "response") else "".join(resp.response_gen)
    finally:
        animation_task.cancel()
        with suppress(asyncio.CancelledError):
            await animation_task

    # ✅ Save response to memory
    if memory:
        memory.put(ChatMessage(role="assistant", content=answer))
        logger.info(f"✅ Assistant reply saved to memory: {answer}")

    answer = extract_and_format_table(answer.strip())
    final = f"✅ นี่คือสิ่งที่พบจาก “{source}”:\n\n{answer}"

    if memory:
        memory.put(ChatMessage(role="assistant", content=final))
        logger.info("🧠 Memory after LLM response:")
        recent_messages = memory.get()[-4:] if memory.get() else []
        for msg in recent_messages:
            logger.info(f"MessageRole.{msg.role.upper()}: {msg.content}")

    # Store the nodes used in the response before returning
    cl.user_session.set("last_answered_context", nodes)

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

# OAuth callback function for OneLogin authentication (only if OAuth is configured)
@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    """Handle OAuth authentication callback from OneLogin."""
    if provider_id == "onelogin":
        logger.info(f"✅ OneLogin OAuth login successful for user: {default_user.display_name}")
        logger.info(f"User groups: {default_user.metadata.get('groups', [])}")
    else:
        logger.info(f"✅ OAuth login successful for provider: {provider_id}, user: {default_user.display_name}")
    
    # Return the user object created by the OAuth provider
    return default_user


# Fallback password auth (enabled when OAuth is not configured)
# Uncomment the decorator below if OAuth is not working and you need password fallback
# @cl.password_auth_callback  
def fallback_password_auth(username: str, password: str):
    """Old password auth - DEPRECATED"""
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
        
    if (username, password) == ("User_31", "123456"):
        logger.info("✅ Login success for User_30")
        return cl.User(
            identifier="User_30",
            metadata={
                "role": "USER",
                "email": "user_30@example.com",
                "provider": "credentials"
            }
        )
    if (username, password) == ("User_32", "123456"):
        logger.info("✅ Login success for User_30")
        return cl.User(
            identifier="User_30",
            metadata={
                "role": "USER",
                "email": "user_30@example.com",
                "provider": "credentials"
            }
        )
        
    if (username, password) == ("User_33", "123456"):
        logger.info("✅ Login success for User_30")
        return cl.User(
            identifier="User_30",
            metadata={
                "role": "USER",
                "email": "user_30@example.com",
                "provider": "credentials"
            }
        )
        
    if (username, password) == ("User_34", "123456"):
        logger.info("✅ Login success for User_30")
        return cl.User(
            identifier="User_30",
            metadata={
                "role": "USER",
                "email": "user_30@example.com",
                "provider": "credentials"
            }
        )

    if (username, password) == ("User_35", "123456"):
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
    return None  # End of old auth function
    


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    """Sets the available chat profiles."""
    return [
        cl.ChatProfile(
            name="Accounting Compliance",
            markdown_description="Got questions about the policy? I'm all ears and ready to help you out—just ask!",
            icon="/public/Gemini_Generated_Image_8jri0n8jri0n8jri (1).svg",
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
    user = cl.user_session.get("user")
    thread_id = cl.context.session.thread_id
    logger.info(f"💬 on_chat_start called for user: {user}")

    # ─── Persist thread row ─────────────────────────────────────
    dl: SQLAlchemyDataLayer = get_data_layer()
    engine = dl.engine
    meta = MetaData()
    threads_table = Table("threads", meta, Column("id", PG_UUID(as_uuid=True), primary_key=True))
    thread_uuid = uuid.UUID(thread_id)
    async with engine.begin() as conn:
        await conn.execute(
            pg_insert(threads_table).values(id=thread_uuid).on_conflict_do_nothing()
        )

    # ─── Setup memory ───────────────────────────────────────────
    redis_session_id = f"{user.identifier}:{thread_id}"
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=TOKEN_LIMIT, chat_store=chat_store, chat_store_key=redis_session_id
    )
    cl.user_session.set("memory", memory)

    # ─── Setup runnable ─────────────────────────────────────────
    setup_runnable()

    # ─── Reset chat profile state (merged from second block) ───
    cl.user_session.set("chat_profile", {
        "name": "default",
        "clarify_state": None,
        "clarification_candidates": [],
        "selected_clarification": None,
        "question_count": 0,
        "current_bu": None  # Optional: reset BU
    })

    # ─── Initialize and store structured chat profile ─────────────
    chat_profile = {
        "name": "default",
        "clarify_state": None,
        "clarification_candidates": [],
        "selected_clarification": None,
        "question_count": 0,
        "current_bu": None  # Optional: reset BU
    }
    cl.user_session.set("chat_profile", chat_profile)

    # ─── Optionally log LLM model if profile name is valid ────────
    profile_name = chat_profile["name"]
    llm_model = CHAT_PROFILES.get(profile_name, {}).get("llm_settings", {}).get("model")
    logger.info(f"Chat started with profile: '{profile_name}', LLM Model ID: '{llm_model}'")

    # ─── Clear clarification state ────────────────────────────────
    clear_clarification_state()

    logger.info("🚀 on_chat_start triggered")

    # ─── Ask BU as first step ─────────────────────────────────────
    await ask_business_unit()


@cl.on_message
async def on_message(message: cl.Message):
    """Handles incoming user messages with natural conversation flow."""
    text = message.content.strip()
    thread_id = cl.context.session.thread_id

    # ✅ Log incoming message
    save_conversation_log(thread_id, message.id, role="user", content=text)

    # ✅ Ensure memory exists and is reused
    thread_id = cl.user_session.get("thread_id") or cl.context.session.thread_id
    user_id = cl.user_session.get("user").identifier
    redis_key = f"{user_id}:{thread_id}"
    logger.info(f"🆔 Thread ID: {thread_id}, User ID: {user_id}, Redis key: {redis_key}")

    memory = cl.user_session.get("memory")
    if memory is None:
        logger.info(f"🧠 Creating new memory with redis_key: {redis_key}")
        try:
            if chat_store is not None:
                memory = ChatMemoryBuffer.from_defaults(
                    token_limit=TOKEN_LIMIT,
                    chat_store=chat_store,
                    chat_store_key=redis_key
                )
                logger.info(f"✅ Memory created with Redis backing")
            else:
                # Use file-based memory fallback
                logger.info("🔄 Using file-based memory fallback")
                memory = ChatMemoryBuffer.from_defaults(token_limit=TOKEN_LIMIT)
                
                # Load existing messages from file if they exist
                if hasattr(globals(), 'use_file_fallback') and use_file_fallback:
                    existing_messages = load_memory_from_file(redis_key)
                    for msg in existing_messages:
                        memory.put(msg)
                    logger.info(f"📂 Loaded {len(existing_messages)} existing messages")
                
            cl.user_session.set("memory", memory)
            cl.user_session.set("memory_key", redis_key)  # Store key for file operations
            logger.info(f"✅ Memory initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create memory: {e}")
            # Fallback to in-memory only
            memory = ChatMemoryBuffer.from_defaults(token_limit=TOKEN_LIMIT)
            cl.user_session.set("memory", memory)
            logger.info("🔄 Using fallback in-memory buffer")
    else:
        logger.info(f"♻️ Using existing memory from session")

    # 💡 Always reassign to Settings.memory to ensure LLM sees the latest buffer
    Settings.memory = memory

    # Check if awaiting business unit selection
    if cl.user_session.get("awaiting_bu_selection", False):
        await handle_bu_selection(text)
        return

    # Check if business unit is selected
    selected_bu = cl.user_session.get("selected_bu")
    if not selected_bu:
        await ask_business_unit()
        return

    # Check if user wants to start a new question/topic (return to BU selection)
    new_question_keywords = [
        "ถามใหม่", "เลือก BU", "เลือก BU ใหม่", "คำถามใหม่", "เปลี่ยนหัวข้อ", "เปลี่ยนหัวข้อใหม่", "เริ่มใหม่", 
        "เลือกหัวข้อใหม่", "หัวข้อใหม่", "เริ่มต้นใหม่","เริ่มใหม่", "เปลี่ยนหมวด", "หมวดใหม่",
        "new question", "new topic", "change topic", "start over", "restart"
    ]
    
    if any(keyword in text.lower() for keyword in new_question_keywords):
        logger.info(f"🔄 User requested new question/topic: {text}")
        # Reset BU selection
        cl.user_session.set("selected_bu", None)
        cl.user_session.set("awaiting_bu_selection", False)
        
        # Clear memory for fresh start
        memory.reset()
        cl.user_session.set("memory", memory)
        
        # Show confirmation and trigger BU selection
        reset_msg = cl.Message(content="✅ เริ่มคำถามใหม่ กรุณาเลือกหัวข้อใหม่")
        await reset_msg.send()
        await ask_business_unit()
        return

    # Add user message to memory
    try:
        memory.put(ChatMessage(role="user", content=text))
        logger.info(f"✅ Added user message to memory: {text[:100]}")
        
        # Debug: show current memory state
        messages = memory.get()
        logger.info(f"🧠 Current memory contents ({len(messages)} total messages):")
        for i, msg in enumerate(messages):
            logger.info(f"  [{i}] {msg.role}: {msg.content[:100]}...")
    except Exception as e:
        logger.error(f"❌ Failed to add user message to memory: {e}")

    # Handle the natural policy question
    await handle_policy_question(text, selected_bu)
    return  # Exit after natural flow - don't continue to old complex logic

    # ─── Global "start new conversation" shortcut ───
    # ─── Global “start new conversation” shortcut ───
    if text == "0" or text == "❌ ถามคำถามใหม่":
        clear_clarification_state()

        for key in [
            "awaiting_clarification",
            PRE_DRILL_KEY,
            AWAITING_PRE_DRILL,
            "pre_drill_nodes",
            "top_k",
            "last_answered_context",
            "last_answered_nodes",
            "nodes",
            "best_node",
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
            "selection_path",   # 🧹 Added this to reset tracked paths
            "node_or_nodes",    # 🧹 Add this if you're storing it in session elsewhere
            "full_text",        # 🧹 Add this
                                     # ✅ Add these for new logic
            "memory",                # 🧠 Reset old memory buffer (optional — depending on your use)
            "clarified_h1",          # 🧩 Clear previous clarified H1
            "drill_level",           # 🔁 Required to exit H1-H2 drill loop
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
        # Wipe Redis-backed memory
        thread_id = cl.context.session.thread_id
        user_id = cl.user_session.get("user").identifier
        redis_key = f"{user_id}:{thread_id}"

        # 🧹 Delete from Redis
        redis_client.delete(redis_key)
        logger.info("🧹 Redis chat store key deleted")

        # 🧠 Clear local reference to memory
        cl.user_session.set("memory", None)

        # 🔄 Re-initialize fresh memory object
        fresh_mem = ChatMemoryBuffer.from_defaults(
            token_limit=TOKEN_LIMIT,
            chat_store=chat_store,
            chat_store_key=redis_key
        )
        cl.user_session.set("memory", fresh_mem)
        logger.info("🧠 Fresh memory buffer initialized")

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


    text = message.content.strip()

    if not runnable or not retriever:
        await send_with_feedback("⚠️ ระบบยังไม่พร้อมใช้งาน กรุณาลองใหม่ภายหลัง")
        return

    # ─── 1. Handle Clarification Input ─────────────────────────────
    if cl.user_session.get("awaiting_clarification"):
        clarification_level = cl.user_session.get("clarification_level", 1)
        logger.info(f"🔄 Awaiting clarification at level: {clarification_level}")

        is_valid = len(text) > 3 and not text.isdigit()
        if is_valid:
            # Check if we just exited H3 clarification - prioritize follow-up detection
            clarification_just_exited = cl.user_session.get("clarification_just_exited")
            last_ctx = cl.user_session.get("last_answered_context")
            
            if clarification_just_exited and last_ctx:
                logger.info("🔍 Post-H3 state detected - bypassing broad question logic to allow follow-up detection")
                # Don't call reset_clarification_state yet - let follow-up detection handle it
                # Fall through to normal message handling which will call handle_followup_or_clarification
                cl.user_session.set("awaiting_clarification", False)  # Exit clarification mode
                # Continue to main message processing below
            else:
                is_broad = await is_broad_but_clear_question_llm(text)
                if is_broad:
                    logger.info("🧠 Broad question detected during clarification — skipping clarification.")
                    reset_clarification_state(text)
                    cl.user_session.set("original_user_question", text)
                    memory.put(ChatMessage(role="user", content=text))
                    return await handle_broad_general_question(text)

                # Otherwise treat it as a valid clarification choice
                return await handle_clarification_response(message, text)
        else:
            # Not valid text but still in clarification, treat as choice
            return await handle_clarification_response(message, text)

    # ─── 2. Continue hierarchical drilldown if in progress ──────────
    if cl.user_session.get("drill_level"):
        return await handle_standard_query(message)

    # ─── 3. Detect fresh general question & update session ──────────
    is_valid = len(text) > 3 and not text.isdigit()
    if is_valid:
        should_skip = await is_broad_but_clear_question_llm(text)
        if should_skip:
            logger.info("🧠 Broad question with no prior context → treat as main question.")
            cl.user_session.set("original_user_question", text)
            return await handle_broad_general_question(text)

    # ─── 4. Backfill original_user_question from pre_drill_query ────
    if cl.user_session.get("original_user_question") is None:
        pre_q = cl.user_session.get("pre_drill_query")
        if pre_q and len(pre_q.strip()) > 3 and not pre_q.strip().isdigit():
            cl.user_session.set("original_user_question", pre_q.strip())
            logger.info(f"📌 Backfilled original_user_question from pre_drill_query = {pre_q.strip()}")
        else:
            logger.info("🚫 No valid pre_drill_query found to backfill original_user_question")

    # ─── 5. Proceed with standard handling ──────────────────────────
    return await handle_standard_query(message)


# ======================================================================================
# Message Handling Logic
# ======================================================================================
def group_by_h2(nodes):
    from collections import defaultdict
    h2_groups = defaultdict(list)
    for n in nodes:
        path = n.node.metadata.get("section_path", [])
        if len(path) >= 2:
            h2_groups[path[1]].append(n)
    return h2_groups

def group_by_h3(nodes):
    from collections import defaultdict
    h3_groups = defaultdict(list)
    for n in nodes:
        path = n.node.metadata.get("section_path", [])
        if len(path) >= 3:
            h3_groups[path[2]].append(n)
    return h3_groups


async def show_h2_options(message):
    raw_h2 = cl.user_session.get("hier_sections", {})
    h2_options = list(raw_h2.keys())

    exit_label = "❌ ถามคำถามใหม่"
    opts = h2_options + [exit_label]

    # ✅ Required for clarification handler to resolve user choice
    cl.user_session.set("clarification_level", 1)
    cl.user_session.set("awaiting_clarification", True)
    cl.user_session.set("hier_sections", {h: raw_h2.get(h, []) for h in h2_options})

    logger.info(f"📋 Final H2s shown to user: {h2_options}")

    lines = [f"{i+1}. {title}" for i, title in enumerate(opts)]
    text = "❓ โปรดเลือกหัวข้อย่อย (ระดับ 2):\n\n" + "\n".join(lines)
    text += "\n\n🔴 ตอบด้วยหมายเลข หรือพิมพ์ชื่อหัวข้อที่ต้องการหรือพิมพ์ '0' เพื่อเริ่มคำถามใหม่"

    await cl.Message(
        content=text,
        author="Customer Service Agent"
    ).send()

async def show_h3_options(message):
    raw_h3 = cl.user_session.get("hier_sections", {})
    h3_options = list(raw_h3.keys())
    
    # Limit to top 4 H3 choices to avoid overwhelming the user
    if len(h3_options) > 4:
        logger.info(f"🔢 Limiting H3 choices from {len(h3_options)} to 4")
        h3_options = h3_options[:4]

    exit_label = "❌ ถามคำถามใหม่"
    opts = h3_options + [exit_label]

    # ✅ Required for clarification handler to resolve user choice
    cl.user_session.set("clarification_level", 2)
    cl.user_session.set("awaiting_clarification", True)
    cl.user_session.set("hier_sections", {h: raw_h3.get(h, []) for h in h3_options})

    logger.info(f"📋 Final H3s shown to user: {h3_options}")

    lines = [f"{i+1}. {title}" for i, title in enumerate(opts)]
    text = "❓ โปรดเลือกหัวข้อย่อย (ระดับ 3):\n\n" + "\n".join(lines)
    text += "\n\n🔴 ตอบด้วยหมายเลข หรือพิมพ์ชื่อหัวข้อที่ต้องการหรือพิมพ์ '0' เพื่อเริ่มคำถามใหม่"

    await cl.Message(
        content=text,
        author="Customer Service Agent"
    ).send()
    
    
    
async def handle_clarification_response(message: cl.Message, text: str):
    """Handles user's response during a clarification flow, including hierarchical clarification."""

    # ─── Hierarchical clarification ───
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

        # Exit option selected
        if idx == len(candidates) - 1:
            cl.user_session.set("awaiting_clarification", False)
            cl.user_session.set("clarification_just_exited", True)
            cl.user_session.set("filtered_nodes", None)
            cl.user_session.set("clarification_level", None)

            # Reset memory
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

            await send_with_feedback("✅ คุณได้เลือกเริ่มต้นคำถามใหม่แล้ว! กรุณาพิมพ์คำถามใหม่ของคุณได้เลย")
            return

        selected_title = titles[idx]
        # Normalize section keys to handle subtle mismatch in H1 titles
        normalized_sections = {k.strip(): v for k, v in sections.items()}
        selected_nodes = normalized_sections.get(selected_title.strip(), [])
        logger.info(f"🎯 Searching hier_sections with key: '{selected_title.strip()}'")
        logger.info(f"🧩 Available section keys: {list(normalized_sections.keys())}")
        clarification_level = cl.user_session.get("clarification_level", 0)
        logger.info(f"🟦 Clarification level: {clarification_level}")
        logger.info(f"🟦 Selected title: {selected_title}")
        logger.info(f"🟦 Nodes under selected title: {len(selected_nodes)}")

        if clarification_level == 0:
            # ✅ H1 selected → group and show H2
            cl.user_session.set("selected_h1", selected_title)

            from collections import defaultdict
            all_nodes = cl.user_session.get("pre_drill_nodes") or []
            raw_h2 = defaultdict(list)
            logger.info(f"🔍 H1 Selection Debug: Looking for H2s under selected_title: '{selected_title}'")
            logger.info(f"🔍 Total nodes in pre_drill_nodes: {len(all_nodes)}")
            
            for n in all_nodes:
                path = n.node.metadata.get("section_path", [])
                if len(path) >= 2:
                    logger.info(f"🔍 Node path[0]: '{path[0]}' vs selected: '{selected_title}' - Match: {path[0] == selected_title}")
                    if path[0] == selected_title:
                        raw_h2[path[1]].append(n)
                        logger.info(f"🔍 ✅ Added H2: '{path[1]}' under H1: '{path[0]}'")
            
            logger.info(f"🔍 Found H2 sections: {list(raw_h2.keys())}")

            if not raw_h2:
                logger.warning(f"🚨 No H2 under H1: {selected_title} — skipping to answer_from_node().")
                clear_clarification_state()
                cl.user_session.set("awaiting_clarification", False)
                orig_q = cl.user_session.get("original_user_question") or message.content.strip()
                matching_chunks = [
                    n for n in all_nodes
                    if len(n.node.metadata.get("section_path", [])) >= 1
                    and n.node.metadata["section_path"][0] == selected_title
                ]
                return await answer_from_node(matching_chunks, orig_q)

            # Check if auto-selection is possible
            if len(raw_h2) == 1:
                # Auto-select the only H2 option
                single_h2 = list(raw_h2.keys())[0]
                logger.info(f"🔄 Auto-selecting single H2: '{single_h2}'")
                
                await cl.Message(
                    content=f"🔄 พบหัวข้อย่อยเพียงหัวข้อเดียว กำลังเลือกอัตโนมัติ: **{single_h2}**",
                    author="Customer Service Agent"
                ).send()
                
                # Set up session for H2 selection
                cl.user_session.set("selected_h2", single_h2)
                selected_h2_nodes = raw_h2[single_h2]
                
                # Continue to H3 logic (same as clarification_level == 1)
                from collections import defaultdict
                raw_h3 = defaultdict(list)
                for n in selected_h2_nodes:
                    path = n.node.metadata.get("section_path", [])
                    if len(path) >= 3 and path[2] and path[1] == single_h2: 
                        raw_h3[path[2]].append(n)

                if not raw_h3:
                    logger.warning(f"🚨 No H3 under auto-selected H2: {single_h2} — answering directly.")
                    clear_clarification_state()
                    cl.user_session.set("awaiting_clarification", False)
                    return await answer_from_node(selected_h2_nodes, user_q=message.content.strip())

                # Check for H3 auto-selection
                logger.info(f"🔍 H3 auto-selection check: raw_h3 has {len(raw_h3)} entries: {list(raw_h3.keys())}")
                if len(raw_h3) == 1:
                    # Auto-select the only H3 option
                    single_h3 = list(raw_h3.keys())[0]
                    logger.info(f"🔄 Auto-selecting single H3: '{single_h3}'")
                    
                    await cl.Message(
                        content=f"🔄 พบหัวข้อย่อยเพียงหัวข้อเดียว กำลังเลือกอัตโนมัติ: **{single_h3}**",
                        author="Customer Service Agent"
                    ).send()
                    
                    # Final answer with auto-selected H3
                    selected_h3_nodes = raw_h3[single_h3]
                    cl.user_session.set("selected_h3", single_h3)
                    cl.user_session.set("filtered_nodes", selected_h3_nodes)
                    clear_clarification_state()
                    cl.user_session.set("awaiting_clarification", False)
                    cl.user_session.set("last_answered_context", selected_h3_nodes)
                    
                    try:
                        # Ensure clarification_just_exited is cleared before proceeding
                        cl.user_session.set("clarification_just_exited", False)
                        logger.info("🎯 H3 auto-selection: calling answer_from_node")
                        await answer_from_node(selected_h3_nodes, user_q=message.content.strip())
                        logger.info("✅ H3 auto-selection: answer_from_node completed successfully")
                        return
                    except Exception as e:
                        logger.error(f"❌ H3 auto-selection: error in answer_from_node: {e}")
                        await cl.Message(f"เกิดข้อผิดพลาดในการประมวลผล: {e}").send()
                        return
                else:
                    # Check for keyword-based auto-selection for specific questions
                    original_question = cl.user_session.get("original_user_question", "").lower()
                    
                    # 1. Check for procurement supplier questions
                    if "procurement" in original_question and ("เปิด" in original_question or "ใหม่" in original_question or "ทำอย่างไร" in original_question):
                        # Look for H3 section containing procurement supplier content
                        procurement_h3 = None
                        for h3_title in raw_h3.keys():
                            # Check if H3 contains procurement supplier info by examining node content
                            h3_nodes = raw_h3[h3_title]
                            for node in h3_nodes[:3]:  # Check first few nodes for efficiency
                                if "คู่ค้าภายใต้การดูแลของแผนกจัดซื้อ" in node.node.text or "procurement supplier" in node.node.text.lower():
                                    procurement_h3 = h3_title
                                    break
                            if procurement_h3:
                                break
                        
                        if procurement_h3:
                            logger.info(f"🔄 Auto-selecting procurement-related H3: '{procurement_h3}' for procurement supplier question")
                            
                            await cl.Message(
                                content=f"🔄 กำลังค้นหาข้อมูลเฉพาะสำหรับคู่ค้า Procurement โดยอัตโนมัติ",
                                author="Customer Service Agent"
                            ).send()
                            
                            # Final answer with auto-selected H3
                            selected_h3_nodes = raw_h3[procurement_h3]
                            cl.user_session.set("selected_h3", procurement_h3)
                            cl.user_session.set("filtered_nodes", selected_h3_nodes)
                            cl.user_session.set("awaiting_clarification", False)
                            cl.user_session.set("clarification_just_exited", True)
                            cl.user_session.set("last_answered_context", selected_h3_nodes)
                            return await answer_from_node(selected_h3_nodes, user_q=message.content.strip())
                    
                    # 2. Check for general document questions
                    elif "เอกสาร" in original_question and ("เปิด" in original_question or "vendor" in original_question or "ใหม่" in original_question):
                        # Look for H3 section about required documents
                        document_h3 = None
                        for h3_title in raw_h3.keys():
                            if "เอกสารที่ต้องใช้" in h3_title and "vendor" in h3_title:
                                document_h3 = h3_title
                                break
                        
                        if document_h3:
                            logger.info(f"🔄 Auto-selecting document-related H3: '{document_h3}' for question about documents")
                            
                            await cl.Message(
                                content=f"🔄 กำลังเลือกส่วนที่เกี่ยวข้องกับเอกสารโดยอัตโนมัติ: **{document_h3}**",
                                author="Customer Service Agent"
                            ).send()
                            
                            # Final answer with auto-selected H3
                            selected_h3_nodes = raw_h3[document_h3]
                            cl.user_session.set("selected_h3", document_h3)
                            cl.user_session.set("filtered_nodes", selected_h3_nodes)
                            cl.user_session.set("awaiting_clarification", False)
                            cl.user_session.set("clarification_just_exited", True)
                            cl.user_session.set("last_answered_context", selected_h3_nodes)
                            return await answer_from_node(selected_h3_nodes, user_q=message.content.strip())
                    
                    # Multiple H3s - show choices
                    cl.user_session.set("clarification_level", 2)
                    cl.user_session.set("awaiting_clarification", True)
                    cl.user_session.set("hier_sections", raw_h3)
                    cl.user_session.set("filtered_nodes", selected_h2_nodes)
                    logger.info(f"📋 Final H3s shown to user: {list(raw_h3.keys())}")
                    return await show_h3_options(message)
            else:
                # Multiple H2s - show choices
                cl.user_session.set("clarification_level", 1)
                cl.user_session.set("awaiting_clarification", True)
                cl.user_session.set("hier_sections", dict(raw_h2))
                cl.user_session.set("filtered_nodes", all_nodes)
                logger.info(f"📋 Final H2s shown to user: {list(raw_h2.keys())}")
                return await show_h2_options(message)

        elif clarification_level == 1:
            # ✅ H2 selected → group and show H3
            cl.user_session.set("selected_h2", selected_title)

            from collections import defaultdict
            raw_h3 = defaultdict(list)
            logger.info(f"🔍 H2 Selection Debug: Looking for H3s under selected H2: '{selected_title}'")
            logger.info(f"🔍 Total nodes in selected_nodes: {len(selected_nodes)}")
            
            for n in selected_nodes:
                path = n.node.metadata.get("section_path", [])
                if len(path) >= 3 and path[2]:
                    logger.info(f"🔍 Node path[1]: '{path[1]}' vs selected: '{selected_title}' - Match: {path[1] == selected_title}")
                    if path[1] == selected_title: 
                        raw_h3[path[2]].append(n)
                        logger.info(f"🔍 ✅ Added H3: '{path[2]}' under H2: '{path[1]}'")
            
            logger.info(f"🔍 Found H3 sections: {list(raw_h3.keys())}")

            if not raw_h3:
                logger.warning(f"🚨 No H3 under H2: {selected_title} — answering directly.")
                clear_clarification_state()
                cl.user_session.set("awaiting_clarification", False)
                return await answer_from_node(selected_nodes, user_q=message.content.strip())

            # Check if auto-selection is possible for H3
            if len(raw_h3) == 1:
                # Auto-select the only H3 option
                single_h3 = list(raw_h3.keys())[0]
                logger.info(f"🔄 Auto-selecting single H3: '{single_h3}'")
                
                await cl.Message(
                    content=f"🔄 พบหัวข้อย่อยเพียงหัวข้อเดียว กำลังเลือกอัตโนมัติ: **{single_h3}**",
                    author="Customer Service Agent"
                ).send()
                
                # Final answer with auto-selected H3
                selected_h3_nodes = raw_h3[single_h3]
                cl.user_session.set("selected_h3", single_h3)
                cl.user_session.set("filtered_nodes", selected_h3_nodes)
                clear_clarification_state()
                cl.user_session.set("awaiting_clarification", False)
                cl.user_session.set("last_answered_context", selected_h3_nodes)
                
                try:
                    # Ensure clarification_just_exited is cleared before proceeding
                    cl.user_session.set("clarification_just_exited", False)
                    logger.info("🎯 H3 auto-selection (manual H2 path): calling answer_from_node")
                    await answer_from_node(selected_h3_nodes, user_q=message.content.strip())
                    logger.info("✅ H3 auto-selection (manual H2 path): answer_from_node completed successfully")
                    return
                except Exception as e:
                    logger.error(f"❌ H3 auto-selection (manual H2 path): error in answer_from_node: {e}")
                    await cl.Message(f"เกิดข้อผิดพลาดในการประมวลผล: {e}").send()
                    return
            else:
                # Check for keyword-based auto-selection for specific questions
                original_question = cl.user_session.get("original_user_question", "").lower()
                
                # 1. Check for procurement supplier questions
                if "procurement" in original_question and ("เปิด" in original_question or "ใหม่" in original_question or "ทำอย่างไร" in original_question):
                    # Look for H3 section containing procurement supplier content
                    procurement_h3 = None
                    for h3_title in raw_h3.keys():
                        # Check if H3 contains procurement supplier info by examining node content
                        h3_nodes = raw_h3[h3_title]
                        for node in h3_nodes[:3]:  # Check first few nodes for efficiency
                            if "คู่ค้าภายใต้การดูแลของแผนกจัดซื้อ" in node.node.text or "procurement supplier" in node.node.text.lower():
                                procurement_h3 = h3_title
                                break
                        if procurement_h3:
                            break
                    
                    if procurement_h3:
                        logger.info(f"🔄 Auto-selecting procurement-related H3: '{procurement_h3}' for procurement supplier question")
                        
                        await cl.Message(
                            content=f"🔄 กำลังค้นหาข้อมูลเฉพาะสำหรับคู่ค้า Procurement โดยอัตโนมัติ",
                            author="Customer Service Agent"
                        ).send()
                        
                        # Final answer with auto-selected H3
                        selected_h3_nodes = raw_h3[procurement_h3]
                        cl.user_session.set("selected_h3", procurement_h3)
                        cl.user_session.set("filtered_nodes", selected_h3_nodes)
                        cl.user_session.set("awaiting_clarification", False)
                        cl.user_session.set("clarification_just_exited", True)
                        cl.user_session.set("last_answered_context", selected_h3_nodes)
                        return await answer_from_node(selected_h3_nodes, user_q=message.content.strip())
                
                # 2. Check for general document questions
                elif "เอกสาร" in original_question and ("เปิด" in original_question or "vendor" in original_question or "ใหม่" in original_question):
                    # Look for H3 section about required documents
                    document_h3 = None
                    for h3_title in raw_h3.keys():
                        if "เอกสารที่ต้องใช้" in h3_title and "vendor" in h3_title:
                            document_h3 = h3_title
                            break
                    
                    if document_h3:
                        logger.info(f"🔄 Auto-selecting document-related H3: '{document_h3}' for question about documents")
                        
                        await cl.Message(
                            content=f"🔄 กำลังเลือกส่วนที่เกี่ยวข้องกับเอกสารโดยอัตโนมัติ: **{document_h3}**",
                            author="Customer Service Agent"
                        ).send()
                        
                        # Final answer with auto-selected H3
                        selected_h3_nodes = raw_h3[document_h3]
                        cl.user_session.set("selected_h3", document_h3)
                        cl.user_session.set("filtered_nodes", selected_h3_nodes)
                        cl.user_session.set("awaiting_clarification", False)
                        cl.user_session.set("clarification_just_exited", True)
                        cl.user_session.set("last_answered_context", selected_h3_nodes)
                        return await answer_from_node(selected_h3_nodes, user_q=message.content.strip())
                
                # No auto-selection possible, show H3 choices
                cl.user_session.set("clarification_level", 2)
                cl.user_session.set("awaiting_clarification", True)
                cl.user_session.set("hier_sections", raw_h3)
                cl.user_session.set("filtered_nodes", selected_nodes)
                logger.info(f"📋 Final H3s shown to user: {list(raw_h3.keys())}")
                return await show_h3_options(message)

        elif clarification_level == 2:
            # ✅ H3 selected → check for H4 or final answer
            cl.user_session.set("selected_h3", selected_title)
            
            # Check if there are H4 sub-sections
            from collections import defaultdict
            raw_h4 = defaultdict(list)
            logger.info(f"🔍 H3 Selection Debug: Looking for H4s under selected H3: '{selected_title}'")
            logger.info(f"🔍 Total nodes in selected_nodes: {len(selected_nodes)}")
            
            for n in selected_nodes:
                path = n.node.metadata.get("section_path", [])
                if len(path) >= 4 and path[3]:
                    logger.info(f"🔍 Node path[2]: '{path[2]}' vs selected: '{selected_title}' - Match: {path[2] == selected_title}")
                    if path[2] == selected_title: 
                        raw_h4[path[3]].append(n)
                        logger.info(f"🔍 ✅ Added H4: '{path[3]}' under H3: '{path[2]}'")
            
            logger.info(f"🔍 Found H4 sections: {list(raw_h4.keys())}")
            
            if not raw_h4:
                # No H4 - final answer
                cl.user_session.set("filtered_nodes", selected_nodes)
                cl.user_session.set("awaiting_clarification", False)
                cl.user_session.set("clarification_just_exited", True)
                cl.user_session.set("last_answered_context", selected_nodes)
                return await answer_from_node(selected_nodes, user_q=message.content.strip())
            
            # Check if auto-selection is possible for H4
            if len(raw_h4) == 1:
                # Auto-select the only H4 option
                single_h4 = list(raw_h4.keys())[0]
                logger.info(f"🔄 Auto-selecting single H4: '{single_h4}'")
                
                await cl.Message(
                    content=f"🔄 พบหัวข้อย่อยเพียงหัวข้อเดียว กำลังเลือกอัตโนมัติ: **{single_h4}**",
                    author="Customer Service Agent"
                ).send()
                
                # Final answer with auto-selected H4
                selected_h4_nodes = raw_h4[single_h4]
                cl.user_session.set("selected_h4", single_h4)
                cl.user_session.set("filtered_nodes", selected_h4_nodes)
                cl.user_session.set("awaiting_clarification", False)
                cl.user_session.set("clarification_just_exited", True)
                cl.user_session.set("last_answered_context", selected_h4_nodes)
                return await answer_from_node(selected_h4_nodes, user_q=message.content.strip())
            else:
                # Multiple H4s - show choices (would need to implement show_h4_options)
                # For now, just provide final answer since H4 choices aren't implemented
                logger.info(f"📋 Found {len(raw_h4)} H4 options, but H4 choice UI not implemented - answering directly")
                cl.user_session.set("filtered_nodes", selected_nodes)
                cl.user_session.set("awaiting_clarification", False)
                cl.user_session.set("clarification_just_exited", True)
                cl.user_session.set("last_answered_context", selected_nodes)
                return await answer_from_node(selected_nodes, user_q=message.content.strip())

    # ─── End hierarchical ───

    clarification_flag = cl.user_session.get("clarification_just_exited")
    last_answered = cl.user_session.get("last_answered_context")
    logger.info(f"🔍 Post-hierarchical check: clarification_just_exited = {clarification_flag}, last_answered = {bool(last_answered)}")
    
    # Only skip if clarification_just_exited is True AND we have successfully answered
    if clarification_flag and last_answered:
        logger.warning("⛔ clarification_just_exited is True and we have answered context — skipping clarification logic")
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
            return await answer_from_node(chosen, message.content.strip())

        await send_with_feedback("⚠️ aaaไม่พบข้อมูลที่เกี่ยวข้อง กรุณาพิมพ์คำถามใหม่")
        clear_clarification_state()
        return

    cl.user_session.set("clarification_rounds", rounds + 1)
    choice = message.content.strip()
    opt_out_label = "❌ ถามคำถามใหม่"

    if (choice.isdigit() and opt_out_label in summaries and
        int(choice) - 1 == summaries.index(opt_out_label)) or \
        choice.strip().lower() in [opt_out_label, "❌", "exit", "ถามคำถามใหม่"]:
        clear_clarification_state()
        cl.user_session.set("clarification_just_exited", True)
        await send_with_feedback("✅ คุณได้เลือกเริ่มต้นคำถามใหม่ กรุณาถามคำถามของคุณอีกครั้ง")
        return

    if choice.lower() == "auto":
        chosen = max(nodes_to_consider, key=lambda n: n.score)
        logger.info(f"[clarify] User requested auto, selecting node with score {chosen.score:.2f}")
        clear_clarification_state()
        return await answer_from_node(chosen, user_q=text)

    selected_index = None
    if choice.isdigit():
        numeric_idx = int(choice) - 1
        if 0 <= numeric_idx < len(summaries):
            selected_index = numeric_idx
        else:
            await send_with_feedback("⚠️ หมายเลขที่คุณเลือกอยู่นอกขอบเขต โปรดลองใหม่")
            return
    else:
        from difflib import SequenceMatcher
        ratios = [SequenceMatcher(None, choice.lower(), s.lower()).ratio() for s in summaries]
        if ratios and max(ratios) > 0.6:
            selected_index = ratios.index(max(ratios))
        else:
            await send_with_feedback("⚠️ ไม่พบหัวข้อที่เลือก โปรดพิมพ์หมายเลขหรือชื่อหัวข้อให้ถูกต้องอีกครั้ง")
            return

    if selected_index is None:
        await send_with_feedback("⚠️ ไม่พบหัวข้อที่เลือก โปรดพิมพ์หมายเลขหรือชื่อหัวข้อให้ถูกต้องอีกครั้ง")
        return

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

    if nodes_to_consider and selected_index < len(nodes_to_consider):
        chosen_node = nodes_to_consider[selected_index]
        await answer_from_node(chosen_node, message.content.strip())
    elif nodes_to_consider:
        chosen_node = max(nodes_to_consider, key=lambda n: n.score)
        await answer_from_node(chosen_node, message.content.strip())
    else:
        await send_with_feedback("⚠️ ไม่พบเนื้อหาที่เกี่ยวข้อง โปรดลองเลือกหัวข้อใหม่หรือลองถามใหม่อีกครั้ง")

    clear_clarification_state()




async def is_broad_but_clear_question_llm(question: str) -> bool:
    """Ask the LLM if the question is a general policy-level question that should skip clarification."""
    llm = get_llm_settings(cl.user_session.get("chat_profile"))
    prompt = (
        f'User asked: "{question}"\n\n'
        "Determine if this is a **broad, general policy-level** question that can be answered directly without needing further clarification.\n\n"
        "✅ Answer 'Yes' if the question is asking for a **definition, general explanation, high-level process overview, policy summary, or general expense/entertainment rules** (e.g., 'DOA คืออะไร', 'LOA ต่างจาก DOA อย่างไร', 'Process ในการสั่งซื้อ ต้องทำอย่างไรบ้าง', 'ขั้นตอนการทำสัญญาคืออะไร', รายชื่อผู้บริหาร, ผู้ที่ต้องติดต่อ, 'พาคู่ค้าไปทานข้าว เบิกค่าใช้จ่ายได้มั้ย', 'ค่าเลี้ยงรับรองเบิกได้อย่างไร', 'entertainment expenses คืออะไร').\n"
        "❌ Answer 'No' if the question includes **specific numbers, exact amounts, particular conditions, detailed scenarios, specific approvals, payment methods, or user-specific logic**.\n\n"
        "Respond with only 'Yes' or 'No'."
    )
    try:
        resp = llm.chat([ChatMessage(role="user", content=prompt)])
        answer = resp.message.content.strip().lower()

        logger.info(f"🧠 [Broad Q Check] LLM response: '{answer}' for question: '{question}'")

        return answer == "yes"
    except Exception as e:
        logger.warning(f"LLM general-question check failed: {e}")
        return False
    
async def show_h1_options(message):
    h1_options = cl.user_session.get("h1_options") or []
    raw_h1 = cl.user_session.get("raw_h1") or {}

    # 🎯 Auto-select if only 1 real choice (excluding exit option)
    if len(h1_options) == 1:
        selected_h1 = h1_options[0]
        logger.info("🎯" * 20)
        logger.info(f"🎯 AUTO-SELECTION LEVEL H1 (ONLY CHOICE)")
        logger.info(f"🎯 Selected: '{selected_h1}'")
        logger.info(f"🎯 Reason: Only 1 option available")
        logger.info("🎯" * 20)
        
        # Simulate the user selecting this H1 option directly
        # Set up the clarification state as if user chose option 1
        cl.user_session.set("clarification_level", 0)
        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("hier_sections", {h: raw_h1.get(h, []) for h in h1_options})
        
        # Create a fake user message selecting the first (and only) option
        from chainlit.message import Message as clMessage
        fake_selection_msg = clMessage(content="1")  # Select the first option
        
        # Process the selection through the clarification handler
        return await on_message(fake_selection_msg)

    exit_label = "❌ ถามคำถามใหม่"
    opts = h1_options + [exit_label]

    # ✅ Required for clarification handler to resolve user choice
    cl.user_session.set("clarification_level", 0)
    cl.user_session.set("awaiting_clarification", True)
    cl.user_session.set("hier_sections", {h: raw_h1.get(h, []) for h in h1_options})

    logger.info("🔍" * 20)
    logger.info(f"🔍 SHOWING H1 CHOICES (NO AUTO-SELECTION)")
    logger.info(f"🔍 Available H1 options: {h1_options}")
    logger.info("🔍" * 20)

    lines = [f"{i+1}. {title}" for i, title in enumerate(opts)]
    text = "❓ โปรดเลือกหัวข้อหลัก (ระดับ 1):\n\n" + "\n".join(lines)
    text += "\n\n🔴 ตอบด้วยหมายเลข หรือพิมพ์ชื่อหัวข้อที่ต้องการหรือพิมพ์ '0' เพื่อเลือก Business Group ใหม่"

    await cl.Message(
        content=text,
        author="Customer Service Agent"
    ).send()

async def handle_followup_or_clarification(message: cl.Message) -> Optional[cl.Message]:
    text = message.content.strip()
    memory = cl.user_session.get("memory")
    last_ctx = cl.user_session.get("last_answered_context")
    clarification_just_exited = cl.user_session.get("clarification_just_exited")

    # Ensure last_ctx is always a list if present
    if last_ctx and not isinstance(last_ctx, list):
        last_ctx = [last_ctx]

    # ── 1) Follow-up check ─────────────────────────────
    followup_answer = "no"
    if memory and memory.get():
        valid_msgs = [
            m for m in memory.get()
            if m.role in {"user", "assistant"}
            and m.content.strip()
            and len(m.content.strip()) > 3
            and not m.content.strip().isdigit()
        ]

        # Do follow-up check if we just exited clarification (and have context) or have enough history
        do_check = False
        clarification_just_exited = cl.user_session.get("clarification_just_exited")
        if clarification_just_exited and last_ctx:
            do_check = True
            logger.info(f"🔍 Follow-up check triggered: clarification_just_exited={clarification_just_exited}, last_ctx_count={len(last_ctx) if last_ctx else 0}")
            logger.info(f"🔍 User message: '{text}' - checking if it's a follow-up to previous H3 selection")
        elif len(valid_msgs) >= 6:
            do_check = True
            logger.info(f"🔍 Follow-up check triggered: enough history ({len(valid_msgs)} messages)")
        else:
            logger.info(f"🔍 Follow-up check skipped: clarification_just_exited={clarification_just_exited}, last_ctx_exists={bool(last_ctx)}, msg_count={len(valid_msgs)}")

        if do_check:
            recent = valid_msgs[-6:] if valid_msgs else []
            context = "\n".join(f"{m.role.title()}: {m.content.strip()}" for m in recent)
            followup_check_prompt = (
                "Given the following chat history and the new user message, "
                "determine if the user is continuing a follow-up from the same topic. "
                "If yes, answer only 'Yes'. If it starts a new topic, answer only 'No'.\n\n"
                f"{context}\nUser: {text}"
            )
            try:
                llm = get_llm_settings(cl.user_session.get("chat_profile"))
                followup_result = llm.chat([ChatMessage(role="user", content=followup_check_prompt)])
                followup_answer = followup_result.message.content.strip().lower()
                logger.info(f"🧠 [Follow-up LLM] → {followup_answer}")
            except Exception as e:
                logger.warning(f"LLM follow-up check failed: {e}")
                followup_answer = "no"
        else:
            logger.info("🧠 Follow-up check skipped — not enough history and not in post-clarification state")
    else:
        logger.info("🧠 Follow-up check skipped — memory is empty")
    followup_answer = followup_answer or "no"

    # ── 2) Follow-up → reuse previous answer context ───
    if followup_answer == "yes" and last_ctx:
        logger.info("🧠 Follow-up detected → reuse last_answered_context")
        if last_ctx:
            logger.info(f"🧩 Reusing last_answered_context with {len(last_ctx)} node(s).")
            # Log the first node for debugging
            if len(last_ctx) > 0:
                first_node = last_ctx[0]
                section_path = first_node.node.metadata.get("section_path", [])
                logger.info(f"🧩 First node section_path: {section_path}")
        if memory:
            memory.put(ChatMessage(role="user", content=text))
        cl.user_session.set("clarification_just_exited", False)
        return await answer_from_node(last_ctx, user_q=text)

    # ── 3) Clarification response handler ───────────────
    if cl.user_session.get("awaiting_clarification"):
        level = cl.user_session.get("clarification_level")
        hier_sections = cl.user_session.get("hier_sections", {})
        choice = message.content.strip()

        exit_label = "❌ ถามคำถามใหม่"
        options = list(hier_sections.keys()) + [exit_label]

        idx = None
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            idx = int(choice) - 1
        elif choice in options:
            idx = options.index(choice)

        if idx is not None:
            selected = options[idx]
            logger.info(f"✅ User selected from clarification level {level}: {selected}")
            cl.user_session.set("awaiting_clarification", False)

            if selected == exit_label:
                cl.user_session.set("clarification_just_exited", True)
                await cl.Message("🌀 เริ่มต้นคำถามใหม่ได้เลยค่ะ").send()
                return None

            if memory:
                memory.put(ChatMessage(role="user", content=f"Clarified: {selected}"))

            # H1 selected
            if level == 1:
                cl.user_session.set("selected_h1", selected)
                cl.user_session.set("drill_level", None)
                from chainlit.message import Message as clMessage
                fake_msg = clMessage(content=selected)
                return await handle_standard_query(fake_msg)

            # H2 or H3
            return await answer_from_node(hier_sections[selected], user_q=text)
        else:
            logger.warning("⚠️ Invalid clarification response")
            await cl.Message("⚠️ หมายเลขที่คุณเลือกอยู่นอกขอบเขต โปรดลองใหม่").send()
            return None  # ←✅ this was missing

    # ── Final fallback: no clarification, no drill ───────────────────────
    if (
        not cl.user_session.get("awaiting_clarification")
        and not cl.user_session.get("drilling")
        and followup_answer == "yes"
    ):
        logger.warning("🛑 Follow-up fallback — no clarification or drill but follow-up was detected.")
        
        # Check if the follow-up contains specific different parameters that need new search
        import re
        amounts = re.findall(r'(\d+)\s*(?:ล้าน|ล้านบาท|million)', text.lower())
        if amounts:
            # Follow-up has specific amounts - needs new search for that amount
            logger.info(f"🔍 Follow-up contains specific amount: {amounts} - performing new search instead of reusing context")
            # Set follow-up mode to get fresh results but maintain context awareness
            cl.user_session.set("in_followup_mode", True)
            return None  # Let normal flow handle with new search
        
        if last_ctx:
            return await answer_from_node(last_ctx, user_q=text)
        else:
            logger.error("🚨 No fallback context (last_ctx) available to answer the follow-up.")
            return await send_with_feedback("ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้")

    return None

async def provide_broad_summary(top_k_nodes, user_q: str):
    """Provide a concise summary for broad questions instead of full detailed answer."""
    
    # Start animated message as a background task
    animation_task = asyncio.create_task(
        send_animated_message(
            base_msg="กำลังเช็ค Policy ให้อยู่ รอสักครู่นะคะ...",
            frames=["🌑","🌒","🌓","🌔","🌕","🌖","🌗","🌘"],
            interval=0.3
        )
    )
    
    try:
        # Get conversation history
        memory = cl.user_session.get("memory")
        conversation_history = ""
        if memory:
            try:
                messages = memory.get()
                logger.info(f"🧠 Total messages in memory for summary: {len(messages)}")
                if messages and len(messages) > 0:
                    formatted_messages = []
                    for msg in messages[-8:]:  # Get last 8 messages for context
                        if msg.role == "user":
                            formatted_messages.append(f"ผู้ใช้: {msg.content}")
                        elif msg.role == "assistant":
                            formatted_messages.append(f"ผู้ช่วย: {msg.content}")
                    conversation_history = "\n".join(formatted_messages)
                    logger.info(f"🧠 Including {len(formatted_messages)} messages in summary context")
            except Exception as e:
                logger.error(f"🧠 Error retrieving memory for summary: {e}")
        
        # Combine the top nodes content and log document sources
        combined_content = ""
        doc_sources = {}
        for i, node in enumerate(top_k_nodes, 1):
            chunk = node.node.text or ""
            combined_content += chunk + "\n\n"
            
            # Track document sources
            source_doc = node.node.metadata.get("source", "Unknown")
            if source_doc not in doc_sources:
                doc_sources[source_doc] = 0
            doc_sources[source_doc] += 1
            
            logger.info(f"📄 Summary node {i}: source={source_doc}, score={getattr(node, 'score', 'N/A'):.3f}")
        
        # Log summary of document sources
        logger.info(f"📊 Summary using nodes from {len(doc_sources)} documents:")
        for doc, count in doc_sources.items():
            logger.info(f"📊   {doc}: {count} nodes")
        
        # Create a summary prompt with conversation history
        llm = get_llm_settings(cl.user_session.get("chat_profile"))
        
        # Build prompt with conversation context if available
        if conversation_history:
            summary_prompt = (
                f'ประวัติการสนทนา:\n{conversation_history}\n\n'
                f'คำถามปัจจุบัน: "{user_q}"\n\n'
                f'เอกสารที่เกี่ยวข้อง:\n{combined_content}\n\n'
                '🎯 คำแนะนำ:\n'
                '- ตอบคำถามโดยพิจารณาบริบทจากประวัติการสนทนา\n'
                '- ใช้ภาษาที่เป็นมิตรและเข้าใจง่าย พร้อมอิโมจิเพื่อความน่าสนใจ 😊\n'
                '- หากเป็นคำถามต่อเนื่อง ให้อ้างอิงถึงสิ่งที่คุยกันมาก่อนหน้า\n'
                '- ตอบกระชับ ชัดเจน และตรงประเด็น\n'
                '- ใช้ภาษาไทยหรือภาษาเดียวกับที่ผู้ใช้ถาม\n\n'
                'คำตอบ:'
            )
        else:
            summary_prompt = (
                f'คำถาม: "{user_q}"\n\n'
                f'เอกสาร:\n{combined_content}\n\n'
                '🎯 โปรดตอบคำถามโดย:\n'
                '- ใช้ภาษาที่เป็นมิตรและเข้าใจง่าย พร้อมอิโมจิที่เหมาะสม 😊\n'
                '- ตอบกระชับและชัดเจน ไม่เกิน 3-4 ประโยค\n'
                '- เน้นประเด็นหลักที่สำคัญ\n'
                '- ใช้ภาษาไทยหรือภาษาเดียวกับที่ผู้ใช้ถาม\n\n'
                'คำตอบ:'
            )
        
        # Let animation run for a bit before making LLM call
        await asyncio.sleep(1.0)
        
        response = llm.chat([ChatMessage(role="user", content=summary_prompt)])
        summary_text = response.message.content.strip()
        
        # Stop the animation
        animation_task.cancel()
        with suppress(asyncio.CancelledError):
            await animation_task
        
        # Send the summary with typewriting effect
        await send_with_feedback(summary_text)
        
        # Add assistant response to memory
        if memory:
            try:
                memory.put(ChatMessage(role="assistant", content=summary_text))
                logger.info(f"✅ Added summary response to memory")
            except Exception as e:
                logger.error(f"❌ Failed to add summary to memory: {e}")
        
        logger.info(f"📝 Provided broad summary for: {user_q}")
        
    except Exception as e:
        # Stop the animation on error
        animation_task.cancel()
        with suppress(asyncio.CancelledError):
            await animation_task
            
        logger.error(f"Error generating broad summary: {e}")
        # Fallback to regular answer if summary fails
        await answer_from_node(top_k_nodes, user_q=user_q)
        
async def handle_broad_general_question(user_q: str):
    """Handles a broad general question by first providing summary, then offering sub-topics if available."""
    from llama_index.core import VectorStoreIndex
    from collections import defaultdict

    logger.info("✅ Broad general question — providing summary first, then checking for sub-topics")

    cl.user_session.set("awaiting_clarification", False)
    cl.user_session.set("clarification_level", None)
    cl.user_session.set("drill_level", None)
    cl.user_session.set("clarification_just_exited", True)
    cl.user_session.set("auto_skipped", True)

    if not user_q.strip().isdigit():
        cl.user_session.set("original_user_question", user_q)
        logger.info(f"📌 Set original_user_question on broad-skip: {user_q}")
    else:
        logger.info(f"🚫 Skipped setting original_user_question on broad-skip: {user_q}")

    # 🧠 Run vector search with more results to find sub-topics
    dataset = DATASET_MAPPING.get(cl.user_session.get("chat_profile"))
    vector_store = qdrant_manager.get_vector_store(dataset, hybrid=True)
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever(similarity_top_k=40)  # Increased to capture distant sections like Non-trade and Trade suppliers
    nodes = retriever.retrieve(user_q)
    
    # Include more nodes for comprehensive summary - use nodes with score >= 0.40 or top 20, whichever is smaller
    high_score_nodes = [n for n in nodes if hasattr(n, 'score') and n.score >= 0.40] if nodes else []
    top_k_for_answer = high_score_nodes[:20] if len(high_score_nodes) <= 20 else nodes[:20]
    top_score = top_k_for_answer[0].score if top_k_for_answer and hasattr(top_k_for_answer[0], "score") else 0.0
    logger.info(f"🔍 Using {len(top_k_for_answer)} nodes for summary (top score = {top_score:.3f})")

    # Save user question in memory
    memory = cl.user_session.get("memory")
    if memory:
        memory.put(ChatMessage(role="user", content=user_q))

    # First, provide a concise summary answer
    if top_k_for_answer:
        await provide_broad_summary(top_k_for_answer, user_q)
        
        # Then check for sub-topics in the broader results - expanded to capture distant supplier sections
        all_nodes = nodes[:35] if nodes else []
        
        # Group by H1 sections to find sub-topics
        h1_groups = defaultdict(list)
        for node in all_nodes:
            path = node.node.metadata.get("section_path", [])
            if len(path) >= 1 and node.score >= 0.35:  # Adjusted for BGE-M3's higher score distribution
                h1_groups[path[0]].append(node)
        
        # Check if we have meaningful sub-topics to offer
        meaningful_h1_groups = {}
        h1_similarity_scores = {}  # Track similarity scores for auto-selection
        user_q_lower = user_q.lower().strip()
        
        # Special handling for supplier questions - skip H1 selection and go directly to H2
        supplier_keywords = ["คู่ค้า", "supplier", "vendor", "ซัพพลายเออร์"]
        is_supplier_question = any(keyword in user_q_lower for keyword in supplier_keywords)
        
        # Filter out H1 topics that are too similar to the user question or not meaningful
        for h1_name, nodes in h1_groups.items():
            h1_lower = h1_name.lower().strip()
            
            # Skip if H1 is too similar to user question (fuzzy match > 80%)
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, user_q_lower, h1_lower).ratio()
            
            # Skip if it's just a definition/meaning topic when user asked broad question
            is_definition_topic = any(word in h1_lower for word in ['ความหมาย', 'คือ', 'definition', 'วัตถุประสงค์'])
            
            # Count significant nodes (score >= 0.35) and highly relevant nodes (score >= 0.50)
            significant_nodes = [n for n in nodes if n.score >= 0.35]
            highly_relevant_nodes = [n for n in nodes if n.score >= 0.50]
            
            # Keep H1 if: not too similar, not definition, and either has multiple significant nodes OR has at least 1 highly relevant node
            has_enough_content = len(significant_nodes) >= 2 or len(highly_relevant_nodes) >= 1
            
            if similarity < 0.80 and not is_definition_topic and has_enough_content:
                meaningful_h1_groups[h1_name] = nodes
                h1_similarity_scores[h1_name] = similarity
                logger.info(f"🔍 Keeping H1 topic: '{h1_name}' (similarity={similarity:.3f}, significant_nodes={len(significant_nodes)}, highly_relevant={len(highly_relevant_nodes)})")
            else:
                logger.info(f"🔍 Skipping H1 topic: '{h1_name}' (similarity={similarity:.3f}, definition={is_definition_topic}, significant_nodes={len(significant_nodes)}, highly_relevant={len(highly_relevant_nodes)})")
        
        # Check for supplier question auto-selection
        supplier_h1_key = None
        for h1_name in meaningful_h1_groups.keys():
            if "Supplier/Vendor" in h1_name or "คู่ค้า" in h1_name:
                supplier_h1_key = h1_name
                break
        
        if is_supplier_question and supplier_h1_key:
            # For supplier questions, automatically select Supplier/Vendor H1 and show H2 choices
            supplier_h1_nodes = meaningful_h1_groups[supplier_h1_key]
            
            # Count H2 sub-topics within Supplier/Vendor H1
            h2_in_supplier = defaultdict(list)
            for node in supplier_h1_nodes:
                path = node.node.metadata.get("section_path", [])
                if len(path) >= 2:
                    h2_in_supplier[path[1]].append(node)
            
            logger.info(f"🔍 [Supplier Q] Auto-selected '{supplier_h1_key}' H1, found {len(h2_in_supplier)} H2 sub-topics")
            for h2_name, h2_nodes in h2_in_supplier.items():
                logger.info(f"🔍 [Supplier Q]   H2: '{h2_name}' ({len(h2_nodes)} nodes)")
            
            if len(h2_in_supplier) > 1:
                # Show H2 choices directly for supplier questions
                h2_options = list(h2_in_supplier.keys())
                cl.user_session.set("selected_h1", "Supplier/Vendor")
                cl.user_session.set("h2_options", h2_options)
                cl.user_session.set("hier_sections", h2_in_supplier)
                cl.user_session.set("pre_drill_nodes", supplier_h1_nodes)
                cl.user_session.set("pre_drill_query", user_q)
                
                await cl.Message(
                    content=f"\n\n📚 พบหัวข้อที่เกี่ยวข้องเพิ่มเติม หากต้องการข้อมูลเฉพาะเจาะจงมากขึ้น:",
                    author="Customer Service Agent"
                ).send()
                
                return await show_h2_options(None)
            else:
                # Only one H2 under Supplier/Vendor, continue to normal logic
                logger.info(f"🔍 [Supplier Q] Only 1 H2 sub-topic, continuing to normal logic")
        
        # Check for H1 auto-selection based on similarity threshold
        H1_AUTO_SELECT_THRESHOLD = 0.5  # Auto-select if similarity > 50% (adjusted for BGE-M3)
        if len(meaningful_h1_groups) > 1:
            # Log all H1 scores for debugging
            logger.info(f"🔍 Found {len(meaningful_h1_groups)} meaningful H1 topics with scores:")
            for h1_name, score in h1_similarity_scores.items():
                logger.info(f"🔍   '{h1_name}': {score:.3f}")
            
            # Find highest scoring H1
            best_h1 = max(h1_similarity_scores.items(), key=lambda x: x[1])
            best_h1_name, best_h1_score = best_h1
            
            # Check BU context before auto-selecting H1
            selected_bu = cl.user_session.get("selected_bu", "")
            is_supplier_bu = "supplier" in selected_bu.lower() or "procurement" in selected_bu.lower() or "คู่ค้า" in selected_bu.lower()
            
            # Override auto-selection if BU context conflicts with best H1
            if is_supplier_bu and "supplier" not in best_h1_name.lower() and "คู่ค้า" not in best_h1_name:
                # Look for Supplier/Vendor H1 in meaningful groups
                supplier_h1 = None
                for h1_name in meaningful_h1_groups:
                    if "supplier" in h1_name.lower() or "คู่ค้า" in h1_name:
                        supplier_h1 = h1_name
                        break
                
                if supplier_h1:
                    logger.info(f"🎯 BU context override: Selecting '{supplier_h1}' instead of '{best_h1_name}' for supplier BU")
                    best_h1_name = supplier_h1
                    best_h1_score = h1_similarity_scores.get(supplier_h1, 0.0)
            
            # Calculate score gap for gap-based auto-selection
            H1_AUTO_SELECT_GAP = 0.08  # Minimum gap for auto-selection
            sorted_h1_scores = sorted(h1_similarity_scores.items(), key=lambda x: x[1], reverse=True)
            score_gap = 0.0
            if len(sorted_h1_scores) >= 2:
                score_gap = sorted_h1_scores[0][1] - sorted_h1_scores[1][1]
                logger.info(f"🔍 H1 score gap: {sorted_h1_scores[0][0]} ({sorted_h1_scores[0][1]:.3f}) vs {sorted_h1_scores[1][0]} ({sorted_h1_scores[1][1]:.3f}) = {score_gap:.3f}")
            
            # Auto-select if either high score OR significant gap
            should_auto_select = (best_h1_score >= H1_AUTO_SELECT_THRESHOLD) or (score_gap >= H1_AUTO_SELECT_GAP and best_h1_score >= 0.35)
            
            if should_auto_select:
                logger.info(f"🎯 H1 auto-selected (high similarity): '{best_h1_name}' (score={best_h1_score:.3f} >= {H1_AUTO_SELECT_THRESHOLD})")
                
                await cl.Message(
                    content=f"🔄 พบหัวข้อที่ตรงกับคำถามมากที่สุด กำลังเลือกอัตโนมัติ: **{best_h1_name}**",
                    author="Customer Service Agent"
                ).send()
                
                # Set up session for H1 auto-selection
                cl.user_session.set("selected_h1", best_h1_name)
                selected_h1_nodes = meaningful_h1_groups[best_h1_name]
                cl.user_session.set("pre_drill_nodes", selected_h1_nodes)
                
                # Continue directly to H2 processing to avoid recursive calls
                h2_groups = defaultdict(list)
                for node in selected_h1_nodes:
                    path = node.node.metadata.get("section_path", [])
                    if len(path) >= 2:
                        h2_groups[path[1]].append(node)
                
                if len(h2_groups) == 1:
                    # Auto-select single H2
                    single_h2 = list(h2_groups.keys())[0]
                    logger.info(f"🔄 Auto-selecting single H2: '{single_h2}'")
                    selected_h2_nodes = h2_groups[single_h2]
                    cl.user_session.set("selected_h2", single_h2)
                    
                    # Continue to H3 processing
                    h3_groups = defaultdict(list)
                    for n in selected_h2_nodes:
                        path = n.node.metadata.get("section_path", [])
                        if len(path) >= 3 and path[2] and path[1] == single_h2: 
                            h3_groups[path[2]].append(n)

                    if not h3_groups:
                        clear_clarification_state()
                        cl.user_session.set("awaiting_clarification", False)
                        return await answer_from_node(selected_h2_nodes, user_q=user_q)

                    if len(h3_groups) == 1:
                        # Auto-select single H3
                        single_h3 = list(h3_groups.keys())[0]
                        logger.info(f"🔄 Auto-selecting single H3: '{single_h3}'")
                        
                        selected_h3_nodes = h3_groups[single_h3]
                        cl.user_session.set("selected_h3", single_h3)
                        cl.user_session.set("filtered_nodes", selected_h3_nodes)
                        clear_clarification_state()
                        cl.user_session.set("awaiting_clarification", False)
                        cl.user_session.set("last_answered_context", selected_h3_nodes)
                        
                        try:
                            cl.user_session.set("clarification_just_exited", False)
                            logger.info("🎯 H3 auto-selection (H1 auto-select path): calling answer_from_node")
                            await answer_from_node(selected_h3_nodes, user_q=user_q)
                            logger.info("✅ H3 auto-selection (H1 auto-select path): answer_from_node completed successfully")
                            return
                        except Exception as e:
                            logger.error(f"❌ H3 auto-selection (H1 auto-select path): error in answer_from_node: {e}")
                            await cl.Message(f"เกิดข้อผิดพลาดในการประมวลผล: {e}").send()
                            return
                    else:
                        # Multiple H3s - show choices
                        cl.user_session.set("clarification_level", 2)
                        cl.user_session.set("awaiting_clarification", True)
                        cl.user_session.set("hier_sections", h3_groups)
                        cl.user_session.set("filtered_nodes", selected_h2_nodes)
                        return await show_h3_options(cl.Message(content=user_q))
                elif len(h2_groups) > 1:
                    # Multiple H2s - show choices
                    cl.user_session.set("clarification_level", 1)
                    cl.user_session.set("awaiting_clarification", True)
                    cl.user_session.set("hier_sections", dict(h2_groups))
                    cl.user_session.set("filtered_nodes", selected_h1_nodes)
                    return await show_h2_options(cl.Message(content=user_q))
                else:
                    # No H2s - answer directly with H1 nodes
                    clear_clarification_state()
                    cl.user_session.set("awaiting_clarification", False)
                    return await answer_from_node(selected_h1_nodes, user_q=user_q)
            
            logger.info(f"🔍 No auto-selection (best score {best_h1_score:.3f} < {H1_AUTO_SELECT_THRESHOLD}) - offering choices")
            
            # Sort H1 options by their highest node score and limit to top 5
            h1_scores = {}
            for h1_name, nodes in meaningful_h1_groups.items():
                # Get the highest score among all nodes in this H1
                max_score = max(node.score for node in nodes) if nodes else 0.0
                h1_scores[h1_name] = max_score
            
            # Sort by score (highest first) and take top 5
            sorted_h1_choices = sorted(h1_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            h1_options = [h1_name for h1_name, _ in sorted_h1_choices]
            
            logger.info(f"🔍 Limited to top {len(h1_options)} H1 choices based on vector scores:")
            for i, (h1_name, score) in enumerate(sorted_h1_choices, 1):
                logger.info(f"🔍   {i}. {h1_name} (max_score: {score:.3f})")
            
            cl.user_session.set("h1_options", h1_options)
            cl.user_session.set("pre_drill_nodes", all_nodes)
            cl.user_session.set("pre_drill_query", user_q)
            
            # Build raw_h1 for the choice handler (only top 5 choices)
            raw_h1 = {h1: meaningful_h1_groups[h1] for h1 in h1_options}
            cl.user_session.set("raw_h1", raw_h1)
            
            # Show additional choices message
            await cl.Message(
                content=f"\n\n📚 พบหัวข้อที่เกี่ยวข้องเพิ่มเติม หากต้องการข้อมูลเฉพาะเจาะจงมากขึ้น:",
                author="Customer Service Agent"
            ).send()
            
            return await show_h1_options(None)
        else:
            # Check for H2 sub-topics within the main H1 (or use first meaningful group if available)
            target_h1 = None
            if meaningful_h1_groups:
                target_h1 = list(meaningful_h1_groups.keys())[0]
                target_nodes = meaningful_h1_groups[target_h1]
            elif h1_groups:
                target_h1 = list(h1_groups.keys())[0]
                target_nodes = h1_groups[target_h1]
            
            if target_h1 and target_nodes:
                h2_groups = defaultdict(list)
                for node in target_nodes:
                    path = node.node.metadata.get("section_path", [])
                    if len(path) >= 2:
                        h2_groups[path[1]].append(node)
                
                # Filter meaningful H2 topics
                meaningful_h2_groups = {}
                for h2_name, nodes in h2_groups.items():
                    h2_lower = h2_name.lower().strip()
                    
                    # Skip if too similar to user question
                    similarity = SequenceMatcher(None, user_q_lower, h2_lower).ratio()
                    significant_nodes = [n for n in nodes if n.score >= 0.35]
                    
                    if similarity < 0.80 and len(significant_nodes) >= 1:
                        meaningful_h2_groups[h2_name] = nodes
                        logger.info(f"🔍 Keeping H2 topic: '{h2_name}' (similarity={similarity:.3f}, nodes={len(significant_nodes)})")
                    else:
                        logger.info(f"🔍 Skipping H2 topic: '{h2_name}' (similarity={similarity:.3f}, nodes={len(significant_nodes)})")
                
                if len(meaningful_h2_groups) > 1:
                    logger.info(f"🔍 Found {len(meaningful_h2_groups)} meaningful H2 sub-topics under '{target_h1}' - offering choices")
                    
                    # Set up for showing H2 options
                    cl.user_session.set("selected_h1", target_h1)
                    cl.user_session.set("clarification_level", 1)
                    cl.user_session.set("awaiting_clarification", True)
                    cl.user_session.set("hier_sections", dict(meaningful_h2_groups))
                    cl.user_session.set("pre_drill_nodes", all_nodes)
                    
                    await cl.Message(
                        content=f"\n\n📚 พบหัวข้อย่อยเพิ่มเติมใน '{target_h1}' หากต้องการข้อมูลเฉพาะเจาะจงมากขึ้น:",
                        author="Customer Service Agent"
                    ).send()
                    
                    return await show_h2_options(None)
            
            logger.info("🔍 No meaningful sub-topics found - answer complete")
    else:
        logger.warning("⚠️ No top_k results available to answer from.")
        return await send_with_feedback("ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในระบบ")

def reset_clarification_state(user_q: str):
    logger.info("🧠 New topic detected → clearing memory and running full drill flow")

    cl.user_session.set("clarification_just_exited", False)
    cl.user_session.set("last_answered_context", None)
    cl.user_session.set("section_path_memory", [])
    cl.user_session.set("pre_drill_nodes", None)  # Optional: Don't reset if numeric choice
    cl.user_session.set("pre_drill_query", None)
    cl.user_session.set("drill_level", None)
    cl.user_session.set("awaiting_clarification", False)
    cl.user_session.set("hier_sections", {})
    cl.user_session.set("original_user_question", user_q)

    # Reset memory buffer
    memory = cl.user_session.get("memory")
    if memory:
        user = cl.user_session.get("user")
        thread_id = cl.context.session.thread_id
        redis_session_id = f"{user.identifier}:{thread_id}"
        new_memory = ChatMemoryBuffer.from_defaults(
            token_limit=TOKEN_LIMIT,
            chat_store=chat_store,
            chat_store_key=redis_session_id
        )
        cl.user_session.set("memory", new_memory)
        
            
async def handle_standard_query(message: cl.Message):
    import re
    from collections import defaultdict
    from difflib import SequenceMatcher
    import statistics

    text = message.content.strip()
    current_q = text
    user_q = text

    # ─── 1. Follow-up or clarification flow ─────────────────────
    followup_result = await handle_followup_or_clarification(message)
    if followup_result:
        return followup_result

    # ─── 2. Broad-but-clear question check ─────────────────────
    should_skip = await is_broad_but_clear_question_llm(user_q)
    if should_skip:
        return await handle_broad_general_question(user_q)

    # ─── 3. Set original question if not in drill/clarify ──────
    already_set = cl.user_session.get("original_user_question") is not None
    drilling = cl.user_session.get("drill_level") is not None
    clarifying = cl.user_session.get("awaiting_clarification")
    looks_like_real_question = len(text) > 3 and not text.isdigit()

    if not already_set and looks_like_real_question and not drilling and not clarifying:
        cl.user_session.set("original_user_question", text)
        logger.info(f"📌 Set new original_user_question = {text}")
    else:
        logger.info(f"📎 Skip setting original_user_question — already set or in clarification/drill mode: {text}")

    # ─── 4. Ensure memory exists ───────────────────────────────
    memory = cl.user_session.get("memory")
    if not memory:
        logger.warning("⚠️ Memory not found. Initializing fallback.")
        user = cl.user_session.get("user")
        thread_id = cl.context.session.thread_id
        redis_session_id = f"{user.identifier}:{thread_id}"
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=TOKEN_LIMIT,
            chat_store=chat_store,
            chat_store_key=redis_session_id
        )
        cl.user_session.set("memory", memory)

    # ─── 5. Build contextual query if needed ───────────────────
    if not clarifying and not drilling:
        filtered_msgs = [
            m for m in memory.get()
            if m.role in {"user", "assistant"}
            and not m.content.strip().isdigit()
            and not m.content.strip().lower().startswith("clarified:")
            and len(m.content.strip()) > 3
        ]
        last_msgs = filtered_msgs[-6:]
        contextual_query = "\n".join(
            [f"{m.role.capitalize()}: {m.content}" for m in last_msgs] + [f"User: {current_q}"]
        )

        logger.info("📌📌📌📌📌📌📌 Full contextual query:")
        for m in last_msgs:
            logger.info(f"MessageRole.{m.role.upper()}: {m.content}")
        logger.info(f"MessageRole.USER (current): {current_q}")

    # ─── 6. Save user input to memory unless in clarification ──
    clarification_just_exited = cl.user_session.get("clarification_just_exited")
    if not clarifying and not clarification_just_exited:
        if memory:
            memory.put(ChatMessage(role="user", content=current_q))
            logger.info(f"🧠 Appended to memory: {current_q}")
        else:
            logger.warning("⚠️ Cannot append to memory — memory is None")
    else:
        logger.info("📌 Skipped appending to memory due to clarification flow")

    # Reset clarification exit flag
    cl.user_session.set("clarification_just_exited", False)

    logger.info("🧠 Memory after input:")
    for m in memory.get():
        logger.info(f"{m.role}: {m.content}")
    logger.info(f"🧠 Memory object ID: {id(memory)}")

    # ─── 7. Log session state ───────────────────────────────────
    original_q = cl.user_session.get("original_user_question")
    logger.info(f"🧩 Session state | original_q={original_q} | clarifying={clarifying} | drilling={drilling}")


    # ─── Decide whether to save this to memory ─────────────────
    clarification_mode = cl.user_session.get("awaiting_clarification")
    clarification_just_exited = cl.user_session.get("clarification_just_exited")

    if not clarification_mode and not clarification_just_exited:
        memory.put(ChatMessage(role="user", content=current_q))
        logger.info(f"🧠 Appendeds to memory: {current_q}")
    else:
        logger.info("📌 Skipped appending to memory due to clarification flow")

    # Always reset clarification exit flag once handled
    cl.user_session.set("clarification_just_exited", False)

    # ─── Debug log memory ──────────────────────────────────────
    logger.info("🧠 Memory after input:")
    for m in memory.get():
        logger.info(f"{m.role}: {m.content}")
    logger.info(f"🧠 Memory object ID: {id(memory)}")

    # ─── Restore clarification vars ─────────────────────────────
    prev_q = cl.user_session.get("pre_drill_query")
    original_q = current_q
    orig_q = current_q
    


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

            # Filter all chunks under selected H1
            doc_nodes = cl.user_session.get("pre_drill_nodes") or []
            filtered_nodes = [
                n for n in doc_nodes
                if (path := n.node.metadata.get("section_path", [])) and len(path) >= 2 and path[0] == selected_h1
            ]
            cl.user_session.set("h1_filtered_nodes", filtered_nodes)

            # 🧠 Ensure original_user_question is cached from pre_drill_query
            if cl.user_session.get("original_user_question") is None:
                pre_q = cl.user_session.get("pre_drill_query")
                if (
                    pre_q
                    and len(pre_q.strip()) > 3
                    and not pre_q.strip().isdigit()
                    and not re.fullmatch(r"^[0-9]+$", pre_q.strip())
                ):
                    cl.user_session.set("original_user_question", pre_q)
                    logger.info(f"📌 Backfilled original_user_question from pre_drill_query = {pre_q}")
                else:
                    logger.info(f"🚫 Skipped backfill of original_user_question — value too short or numeric: {pre_q}")
            else:
                logger.info(f"📎 original_user_question already set = {cl.user_session.get('original_user_question')}")

            return await handle_standard_query(message)

        else:
            # 🔒 Only set original_user_question if not in H1 drill and no valid selection
            text = message.content.strip()
            is_valid_question = (
                len(text) > 3
                and not text.isdigit()
                and not re.fullmatch(r"^[0-9]+$", text)
            )

            if cl.user_session.get("original_user_question") is None and is_valid_question:
                cl.user_session.set("original_user_question", text)
                logger.info(f"📌 Set new original_user_question = {text}")
            else:
                logger.info(f"📎 original_user_question already set or input invalid: {cl.user_session.get('original_user_question')} | input = {text}")

            # 🧠 Fallback: treat it as a new question if it looks like one
            if len(user_input) > 10 and not user_input.isdigit():
                logger.info("🧠 User likely typed a real question, exiting H1 drill mode.")
                cl.user_session.set("drill_level", None)
                cl.user_session.set("h1_options", None)
                return await handle_standard_query(message)

            await cl.Message("❌ ไม่พบหัวข้อที่คุณเลือก โปรดลองอีกครั้ง หรือพิมพ์ 0 เพื่อเริ่มใหม่").send()
            return
    
     # ─── 11) Fuzzy-fallback & final LLM answer ─────────────────────────────────
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
        
        # Inside handle_standard_query
        # ─── Determine query to use: always contextual if not in clarification/drill ───
        current_q = message.content.strip()
        in_clarification = cl.user_session.get("awaiting_clarification")
        in_drill = cl.user_session.get("drill_level")

        query_to_use = current_q  # default fallback

        if not in_clarification and not in_drill:
            memory = cl.user_session.get("memory")
            if memory:
                filtered_msgs = [
                    m for m in memory.get()
                    if m.role in {"user", "assistant"}
                    and not m.content.strip().isdigit()
                    and not m.content.strip().lower().startswith("clarified:")
                    and len(m.content.strip()) > 3
                ]
                last_msgs = filtered_msgs[-6:]
                contextual_query = "\n".join(
                    [f"{m.role.capitalize()}: {m.content}" for m in last_msgs] + [f"User: {current_q}"]
                )

                logger.info("📌📌📌📌📌📌📌 Full contextual query:")
                for m in last_msgs:
                    logger.info(f"MessageRole.{m.role.upper()}: {m.content}")
                logger.info(f"MessageRole.USER (current): {current_q}")

                query_to_use = contextual_query
                logger.info("🫡🫡🫡🫡 Using contextual_query for retrieval")

        # ─── Run retrieval ───
        all_nodes = pre_drill_retriever.retrieve(query_to_use)
        cl.user_session.set("pre_drill_nodes", all_nodes)
        logger.info(f"🧠🫡📌🧠🫡📌🧠🫡📌 Saved {len(all_nodes)} nodes to session for drill-down")
        # Set flag to use contextual_query next time
        cl.user_session.set("used_contextual_query", True)

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
        # 🔍 Log H1 nodes and their scores
        logger.info("🧠🧠🧠🧠 Top nodes and their H1 section scores:")
        for n in all_nodes:
            path = n.node.metadata.get("section_path", [])
            h1 = path[0] if len(path) > 0 else "❓ Missing H1"
            score = n.score if hasattr(n, "score") else 0.0
          

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
        BU_RELEVANCE_THRESHOLD = 0.45
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
        
        

        if policy_score >= POLICY_AUTO_THRESH:
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
        text = message.content.strip()

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

                logger.info("📚 No H3 found under H2: %s → %d chunks sent", selected, len(matching_chunks))
                return await answer_from_node(matching_chunks, user_q=text)

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
            logger.info("🧠 Current chat memory state: %s", dict(cl.user_session._data))
            return

        # H3 → answer immediately
        else:  # level == 3
            h3_nodes = hier[selected]
            logger.info(f"✅ H3 selected: “{selected}” → {len(h3_nodes)} chunks sent via answer_from_nodes")
            clear_clarification_state()
            cl.user_session.set("awaiting_clarification", False)
            latest_user_q = current_q
            return await answer_from_node(h3_nodes, user_q=text)



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
            from difflib import SequenceMatcher
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
                    return await answer_from_node(best_node, user_q=text)

        except Exception:
            logger.exception("❌ Retrieval failed")
            await send_with_feedback("⚠️ เกิดข้อผิดพลาดในการค้นหา กรุณาลองใหม่อีกครั้ง")
            return

    # ─── 6) Scores, early-reject, auto-drill & auto-answer ─────────────────────
    # ─── 6) Scores, early-reject, auto-drill & auto-answer ─────────────────────
    # ✅ NEW: Reset pre_drill_nodes for follow-up with fresh vector results
    if cl.user_session.get("in_followup_mode"):
        logger.info("🔁 Follow-up mode active — setting new pre_drill_nodes from retrieved vector nodes.")
        cl.user_session.set("pre_drill_nodes", nodes)
        
        # Try to maintain context from previous H3 selection if available
        prev_h3 = cl.user_session.get("selected_h3")
        if prev_h3:
            logger.info(f"🎯 Previous H3 context: '{prev_h3}' - trying to auto-navigate to same section")
            # Filter nodes to same H3 section type if found in results
            filtered_for_h3 = [
                n for n in nodes 
                if (path := n.node.metadata.get("section_path", [])) and len(path) >= 3 
                and any(keyword in path[2].lower() for keyword in ['อนุมัติ', 'capex', 'งบประมาณ', 'โครงการ'] if 'อนุมัติ' in prev_h3.lower())
            ]
            if filtered_for_h3:
                logger.info(f"🎯 Found {len(filtered_for_h3)} nodes in similar H3 context - using filtered set")
                cl.user_session.set("pre_drill_nodes", filtered_for_h3)
        
        # Clear follow-up mode flag after processing
        cl.user_session.set("in_followup_mode", False)
    else:
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
    DEEP_DIRECT_THRESHOLD = 0.35  # Adjusted for BGE-M3's higher score distribution
    DEEP_GAP_THRESHOLD    = 0.060  # Adjusted for BGE-M3's higher score distribution

    # look at your full pre‐drill to see how deep your document actually goes
    all_pre_drill = cl.user_session.get("pre_drill_nodes") or []
    logger.info(f"📦 Retrieved {len(all_pre_drill)} pre-drill nodes")
    for idx, node in enumerate(all_pre_drill[:5]):
        path = node.node.metadata.get('section_path', [])
        score = getattr(node, 'score', 'N/A')
        logger.info(f"🔍 Node {idx+1}: path={path}, score={score}")


    # Safety check for empty sequence
    if not all_pre_drill:
        logger.warning("⚠️ all_pre_drill is empty - cannot calculate max_depth")
        max_depth = 0
        target_depth = 0
    else:
        max_depth = max(len(n.node.metadata.get("section_path", [])) for n in all_pre_drill)
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

            # Check if this is a D&B question that spans both Trade and Non-trade sections
            query_lower = text.lower()
            is_db_question = any(keyword in query_lower for keyword in ["d&b", "dun", "bradstreet", "ประเมินความเสี่ยง"])
            
            all_nodes = cl.user_session.get("h1_filtered_nodes") or cl.user_session.get("pre_drill_nodes") or []
            
            if is_db_question:
                # For D&B questions, include both Trade and Non-trade D&B sections
                matching_section = [
                    n for n in all_nodes
                    if len(n.node.metadata.get("section_path", [])) >= 3 and (
                        "d&b" in n.node.metadata["section_path"][2].lower() or
                        "dun" in n.node.metadata["section_path"][2].lower() or
                        "bradstreet" in n.node.metadata["section_path"][2].lower() or
                        "ประเมินความเสี่ยง" in n.node.metadata["section_path"][2].lower()
                    )
                ]
                logger.info(f"🏷 D&B question detected: including {len(matching_section)} D&B-related nodes from both Trade and Non-trade sections")
            else:
                # Check if question spans multiple categories (CAPEX, OPEX, etc.)
                query_lower = text.lower()
                spans_multiple_categories = any([
                    "capex" in query_lower and "opex" in query_lower,
                    ("capex" in query_lower or "ลงทุน" in query_lower) and ("opex" in query_lower or "ดำเนินงาน" in query_lower),
                    "และ" in query_lower and any(cat in query_lower for cat in ["capex", "opex", "ลงทุน", "ดำเนินงาน"])
                ])
                
                best_path = best_node.node.metadata.get("section_path", [])
                
                if spans_multiple_categories and len(best_path) >= 1:
                    # For questions spanning multiple categories, use H1 level to capture both CAPEX and OPEX
                    parent_section = best_path[:1]
                    matching_section = [
                        n for n in all_nodes
                        if len(n.node.metadata.get("section_path", [])) >= 1 and 
                           n.node.metadata["section_path"][:1] == parent_section
                    ]
                    logger.info(f"🏷 Multi-category question: using all nodes under H1 section: {' / '.join(parent_section)} ({len(matching_section)} nodes)")
                elif len(best_path) >= 2:
                    # Single category: use H2 level to capture all related subsections
                    parent_section = best_path[:2]
                    matching_section = [
                        n for n in all_nodes
                        if len(n.node.metadata.get("section_path", [])) >= 2 and 
                           n.node.metadata["section_path"][:2] == parent_section
                    ]
                    logger.info(f"🏷 Using all nodes under H2 section: {' / '.join(parent_section)} ({len(matching_section)} nodes)")
                else:
                    # Fallback: use exact section_path match
                    matching_section = [
                        n for n in all_nodes
                        if n.node.metadata.get("section_path", []) == best_path
                    ]
                    logger.info(f"🏷 Using exact section match: {' / '.join(best_path)} ({len(matching_section)} nodes)")
                
                # Sort by score and limit to prevent context overflow  
                matching_section = sorted(matching_section, key=lambda x: getattr(x, 'score', 0), reverse=True)[:25]

            clear_clarification_state()
            cl.user_session.set("awaiting_clarification", False)
            return await answer_from_node(matching_section, user_q=text)
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
    VECTOR_AUTO_LEVEL3_THRESHOLD = 0.7  # Adjusted for BGE-M3's higher score distribution
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
                h3_chunks = [
                    n for n in all_nodes
                    if len(n.node.metadata.get("section_path", [])) >= 3
                    and n.node.metadata["section_path"][1] == top_h2
                ]

                if h3_chunks:
                    from collections import defaultdict
                    raw_h3 = defaultdict(list)
                    for n in h3_chunks:
                        path = n.node.metadata.get("section_path", [])
                        raw_h3[path[2]].append(n)

                    logger.info("🏷 Auto-drill into H3 for '%s' → %d chunks", top_h2, len(h3_chunks))

                    cl.user_session.set("awaiting_clarification", True)
                    cl.user_session.set("clarification_level", 2)
                    cl.user_session.set("hier_sections", dict(raw_h3))
                    cl.user_session.set("filtered_nodes", h3_chunks)
                    cl.user_session.set("selected_h2", top_h2)

                    return await show_h3_options(message)

    # 6c) Auto‐answer if extremely confident
    VECTOR_AUTO_DIRECT_THRESHOLD = 0.73  # Adjusted for BGE-M3's higher score distribution 
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
        return await answer_from_node(matching_section, user_q=text)


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
    # 🔧 Simple logic: If high-scoring content exists under different parent levels, show choices
    HIGH_SCORE_THRESHOLD = 0.53  # Adjusted for BGE-M3's higher score distribution
    
    # Group high-scoring nodes by H1 parent
    h1_high_score_groups = defaultdict(list)
    section_scores = defaultdict(float)
    
    for n in all_doc_nodes:
        path = n.node.metadata.get("section_path", [])
        score = getattr(n, "score", 0.0)
        
        if len(path) >= 1:
            h1 = path[0]
            section_scores[h1] = max(section_scores[h1], score)
            
            # Track high-scoring content
            if score >= HIGH_SCORE_THRESHOLD:
                h1_high_score_groups[h1].append((n, score))
    
    # Check if multiple H1 sections have high-scoring content
    h1_with_high_scores = [h1 for h1, group in h1_high_score_groups.items() if group]
    multiple_good_h1s = len(h1_with_high_scores) > 1
    
    # Limit to top 4 highest-scoring H1 sections to avoid overwhelming user
    if multiple_good_h1s and len(h1_with_high_scores) > 4:
        # Sort by max score and take top 4
        h1_scores = [(h1, section_scores[h1]) for h1 in h1_with_high_scores]
        h1_scores.sort(key=lambda x: x[1], reverse=True)
        h1_with_high_scores = [h1 for h1, _ in h1_scores[:4]]
        logger.info(f"🔍 Limited to top 4 H1 sections: {h1_with_high_scores}")
    
    logger.info(f"🔍 H1 sections with high scores (>={HIGH_SCORE_THRESHOLD}): {h1_with_high_scores}")
    logger.info(f"🔍 Multiple good H1 sections: {multiple_good_h1s}")
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

    # H1 Auto-selection thresholds
    H1_AUTO_SELECT_THRESHOLD = 0.53  # Adjusted for BGE-M3's higher score distribution
    H1_AUTO_SELECT_GAP = 0.08        # Minimum gap for auto-selection
    H1_FUZZY_MATCH_THRESHOLD = 0.85  # Minimum fuzzy match similarity for auto-selection

    # Check for fuzzy match between user input and H1 choices
    from difflib import SequenceMatcher
    user_input_clean = current_q.strip().lower()
    best_fuzzy_match = None
    best_fuzzy_score = 0.0
    
    logger.info(f"🔍 Checking fuzzy match for user input: '{current_q.strip()}'")
    for h1_name, h1_score in ordered_h1:
        h1_clean = h1_name.strip().lower()
        fuzzy_score = SequenceMatcher(None, user_input_clean, h1_clean).ratio()
        logger.info(f"🔍 H1 '{h1_name}' fuzzy similarity: {fuzzy_score:.3f}")
        
        if fuzzy_score > best_fuzzy_score:
            best_fuzzy_score = fuzzy_score
            best_fuzzy_match = h1_name
    
    logger.info(f"🔍 Best fuzzy match: '{best_fuzzy_match}' with score {best_fuzzy_score:.3f} (threshold: {H1_FUZZY_MATCH_THRESHOLD})")
    
    # Auto-select if fuzzy match exceeds threshold
    if best_fuzzy_score >= H1_FUZZY_MATCH_THRESHOLD:
        logger.info(f"🎯 H1 auto-selected (fuzzy match): '{best_fuzzy_match}' (similarity={best_fuzzy_score:.3f})")
        cl.user_session.set("selected_h1", best_fuzzy_match)
        # Continue to H2 logic below
    elif len(ordered_h1) == 1:
        # Only one H1 available - auto-select if score is high enough
        top_h1, top_score = ordered_h1[0]
        if top_score >= H1_AUTO_SELECT_THRESHOLD:
            logger.info(f"🎯 H1 auto-selected (only option): '{top_h1}' (score={top_score:.3f})")
            cl.user_session.set("selected_h1", top_h1)
            # Continue to H2 logic below
        else:
            logger.info(f"🔍 Single H1 score too low ({top_score:.3f} < {H1_AUTO_SELECT_THRESHOLD}) - showing options")
            cl.user_session.set("drill_level", "h1")
            cl.user_session.set("h1_options", [top_h1])
            cl.user_session.set("pre_drill_query", current_q)
            cl.user_session.set("pre_drill_nodes", all_doc_nodes)
            
            if (
                cl.user_session.get("original_user_question") is None
                and len(current_q.strip()) > 3
                and not current_q.strip().isdigit()
                and not re.fullmatch(r"^[0-9]+$", current_q.strip())
                and not current_q.lower().startswith("clarified:")
            ):
                cl.user_session.set("original_user_question", current_q)
                logger.info(f"📌 Set original_user_question = {current_q}")
            
            return await show_h1_options(message)
    
    elif len(ordered_h1) >= 2:
        top_h1, top_score = ordered_h1[0]
        second_h1, second_score = ordered_h1[1]
        score_gap = top_score - second_score
        logger.info("🔍 H1 score gap = %.3f", score_gap)

        # Force H1 choices if multiple sections have high-scoring content, UNLESS there's a very clear winner
        if multiple_good_h1s and not (top_score >= 0.7 and score_gap >= 0.06):
            logger.info(f"🎯 Multiple H1 sections with high scores - forcing user choice")
            cl.user_session.set("drill_level", "h1")
            
            # Show only H1s that have high-scoring content
            filtered_h1s = h1_with_high_scores
            cl.user_session.set("h1_options", filtered_h1s)
            cl.user_session.set("pre_drill_query", current_q)
            cl.user_session.set("pre_drill_nodes", all_doc_nodes)
            
            if (
                cl.user_session.get("original_user_question") is None
                and len(current_q.strip()) > 3
                and not current_q.strip().isdigit()
                and not re.fullmatch(r"^[0-9]+$", current_q.strip())
                and not current_q.lower().startswith("clarified:")
            ):
                cl.user_session.set("original_user_question", current_q)
                logger.info(f"📌 Set original_user_question = {current_q}")
            
            return await show_h1_options(message)
        # Auto-select H1 if high confidence and clear winner AND no multiple good sections
        # More aggressive auto-selection for very high scores
        elif (top_score >= H1_AUTO_SELECT_THRESHOLD and score_gap >= H1_AUTO_SELECT_GAP) or (top_score >= 0.7 and score_gap >= 0.04):
            auto_reason = "high score + gap" if top_score >= H1_AUTO_SELECT_THRESHOLD else "very high score + small gap"
            logger.info(f"🎯 H1 auto-selected ({auto_reason}): '{top_h1}' (score={top_score:.3f}, gap={score_gap:.3f})")
            cl.user_session.set("selected_h1", top_h1)
            # Continue to H2 lcanogic below instead of showing H1 options
        elif score_gap < 0.08:  # not a big gap, means ambiguity
            cl.user_session.set("drill_level", "h1")

            # Filter H1s with score > 0.42 (aligned with HIGH_SCORE_THRESHOLD)
            filtered_h1s = [h1 for h1, score in ordered_h1 if score > 0.42]

            # Fallback: if none match, force top 3
            if not filtered_h1s:
                filtered_h1s = [h1 for h1, _ in ordered_h1[:3]]

            # ❗ Ensure this is what gets used
            cl.user_session.set("h1_options", filtered_h1s)
            cl.user_session.set("pre_drill_query", current_q)
            cl.user_session.set("pre_drill_nodes", all_doc_nodes)
            
            # ✅ Set original_user_question ONLY IF not set and this is a real question
            if (
                cl.user_session.get("original_user_question") is None
                and len(current_q.strip()) > 3
                and not current_q.strip().isdigit()
                and not re.fullmatch(r"^[0-9]+$", current_q.strip())
                and not current_q.lower().startswith("clarified:")
            ):
                cl.user_session.set("original_user_question", current_q)
                logger.info(f"📌 Set original_user_question = {current_q}")
            else:
                logger.info(f"🛑 Skipped overwriting original_user_question with: {current_q}")

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
            return await answer_from_node(h2_nodes, user_q=text)

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
    # Only reset if we're NOT in the middle of a clarification flow
    if not cl.user_session.get("awaiting_clarification"):
        logger.info("🧹 Resetting session state after completing answer")
        clear_clarification_state()

        # 🧹 Clear all related session state  
        for key in [
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
            if key in (PRE_DRILL_KEY, AWAITING_PRE_DRILL, "awaiting_bu_selection"):
                cl.user_session.set(key, False)
            else:
                cl.user_session.set(key, None)
    else:
        logger.info("🔄 Preserving clarification state - awaiting user choice")

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

    # Include prior messages from memory (last 3 user-assistant pairs)
    memory = cl.user_session.get("memory")
    prior_messages = memory.get()

    # Grab last 3 pairs = 6 messages max (assuming U-A-U-A-U-A)
    dialogue = prior_messages[-6:]
    history_snippets = ""
    for m in dialogue:
        role = "👤 ผู้ใช้" if m.role == "user" else "🤖 ผู้ช่วย"
        history_snippets += f"{role}: {m.content.strip()}\n"

    # Get the final user message to display as key intent
    last_user_msg = next((m.content for m in reversed(dialogue) if m.role == "user"), "")

    logger.info("🧠 Chat Memory Used in Prompt:")
    for m in dialogue:
        logger.info(f"{m.role}: {m.content.strip()}")

    context_str = (
        f"📜 ประวัติการสนทนา:\n{history_snippets.strip()}\n\n"
        f"📌 คำถามหลักจากผู้ใช้: \"{last_user_msg}\"\n\n"
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

    last_user_msg = next((m.content for m in reversed(prior_messages) if m.role == "user"), query)
    filtered_query = (
        f'📌 คำถามหลักจากผู้ใช้: "{last_user_msg}"\n\n'
        "กรุณาจัดทำคำตอบโดยอ้างอิงจากเนื้อหาทั้งหมดที่ให้ไว้ด้านล่างนี้อย่างครบถ้วนและถูกต้อง:\n\n"
        f"{context_str}"
        f"{formatting_hint}"
        f"{constraint}\n\n"
        "โปรดให้คำตอบอย่างชัดเจน ถูกต้อง และเป็นทางการ โดยระบุชื่อเอกสารนโยบายหรือแหล่งอ้างอิงที่ใช้ในการตอบทุกครั้ง\n"
        "หากไม่พบข้อมูลเพียงพอในเนื้อหาที่ให้ไว้ หรือมีความไม่แน่ใจ กรุณาตอบกลับเพื่อขอข้อมูลเพิ่มเติมจากผู้ใช้งานก่อนตอบ"
        "\n\n⛔ ห้ามสร้างคำตอบจากความเข้าใจส่วนตัวหรือข้อมูลภายนอกเด็ดขาด หากไม่มีข้อมูลในเนื้อหาที่ให้ไว้ ให้ขอข้อมูลเพิ่มเติมแทน"
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