# Run application locally using this command: chainlit run app.py -h --root-path /chatbot/v1
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.llms.groq import Groq
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from auth.onelogin_oauth_provider import OneLoginOAuthProvider
from auth.inject_custom_auth import add_custom_oauth_provider
from functions.qdrant_vectordb import QdrantManager
from chainlit.types import ThreadDict
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
from typing import Dict, Optional
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT
import chainlit as cl
import os
import logging
import warnings
warnings.filterwarnings("ignore")
# By pass SSL certification
import httpx
# Apply the monkey patch
from patches import patch
patch.apply_patch()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load variables from the .env file
load_dotenv()
# Access the variables
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")
API_KEY_CHATBOT = os.getenv("API_KEY_CHATBOT")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID")
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Collection Name (Vector database)
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Chat-Memory
REDIS_CHATSTORE_URI = os.getenv("REDIS_CHATSTORE_URI")
REDIS_CHATSTORE_PASSWORD = os.getenv("REDIS_CHATSTORE_PASSWORD")
# Chat-Memory Token Limit
TOKEN_LIMIT = 1024

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

add_custom_oauth_provider("onelogin", OneLoginOAuthProvider())

chat_store = RedisChatStore(
    redis_url=REDIS_CHATSTORE_URI, db=0, password=REDIS_CHATSTORE_PASSWORD, ttl=180
)

# Set the desired chunk size and context window
# Settings.chunk_size = 512
Settings.context_window = 6000

# Set up embedding model
Settings.embed_model = TextEmbeddingsInference(
    model_name=EMBED_MODEL_ID,
    base_url=EMBED_BASE_URL,
    auth_token=f"Bearer {API_KEY_CHATBOT}",
    timeout=60,
    embed_batch_size=10,
)

# Initialize QdrantManager (preload model)
qdrant_manager = QdrantManager()
vector_store = qdrant_manager.get_vector_store(COLLECTION_NAME, hybrid=True)
index = VectorStoreIndex.from_vector_store(vector_store)

def get_current_chainlit_thread_id() -> str:
    return cl.context.session.thread_id

# Constants and configurations
DATASET_MAPPING = {
    "Lotus's AI": COLLECTION_NAME,
    "Qwen-2.5-32b": COLLECTION_NAME,  # Adjust based on actual configuration
}

CHAT_ENGINE_PARAMS = {
    'chat_mode': "condense_plus_context",
    'similarity_top_k': 5,
    'sparse_top_k': 12,
    'alpha': 0.5,
    # 'vector_store_query_mode': 'hybrid'
}

CHAT_PROFILES = {
    "Lotus's AI": {
        "context_prompt": SYSTEM_PROMPT,
        "welcome_message": "สวัสดีค่ะคุณ {firstname} วันนี้มีอะไรให้น้องบัวช่วยเหลือบ้างคะ",
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
    "Qwen-2.5-32b": {
        "context_prompt": SYSTEM_PROMPT,
        "welcome_message": "สวัสดีค่ะคุณ {firstname} วันนี้มีอะไรให้น้องบัวช่วยเหลือบ้างคะ",
        "llm_settings": {
            "model": GROQ_MODEL_ID,
            "api_key": GROQ_API_KEY,
            "is_chat_model": True,
            "is_function_calling_model": False,
            "temperature": 0.7,
        },
    },
}

def load_context_prompt(chat_profile: str) -> str:
    """Load the context prompt for the given chat profile."""
    return CHAT_PROFILES.get(chat_profile, {}).get("context_prompt", "")

def get_llm_settings(chat_profile: str):
    """
    Retrieve and configure LLM settings based on the chat profile.

    Args:
        chat_profile (str): The name of the chat profile to retrieve settings for.

    Returns:
        LLM: The configured LLM settings.

    Raises:
        ValueError: If no LLM settings are found for the given chat profile,
            or if the chat profile is not supported.
    """
    settings = CHAT_PROFILES.get(chat_profile, {}).get("llm_settings")
    if not settings:
        raise ValueError(f"No LLM settings found for profile: {chat_profile}")
    
    if chat_profile == "Lotus's AI":
        return OpenAILike(
            model=settings["model"],
            api_base=settings["api_base"],
            api_key=settings["api_key"],
            is_chat_model=settings["is_chat_model"],
            is_function_calling_model=settings["is_function_calling_model"],
            temperature=settings["temperature"],
            http_client=settings["http_client"],
        )
    elif chat_profile == "Qwen-2.5-32b":
        return Groq(
            model=settings["model"],
            api_key=settings["api_key"],
            is_chat_model=settings["is_chat_model"],
            is_function_calling_model=settings["is_function_calling_model"],
            temperature=settings["temperature"],
        )
    else:
        raise ValueError(f"Unsupported chat profile: {chat_profile}")

# Example usage in create_chat_engine
def create_chat_engine(chat_profile: str, memory: ChatMemoryBuffer):
    """
    Create a chat engine based on the specified chat profile.

    Args:
        chat_profile (str): The name of the chat profile to create the chat engine for.
        memory (ChatMemoryBuffer): The chat memory buffer to be used by the chat engine.

    Returns:
        A chat engine instance if successful, None otherwise.
    """
    logger.info("Creating chat engine for profile: %s", chat_profile)
    
    context_prompt = load_context_prompt(chat_profile)
    
    dataset = DATASET_MAPPING.get(chat_profile)
    if not dataset:
        logger.error(f"No dataset configured for profile: {chat_profile}")
        return None
    
    try:
        vector_store = qdrant_manager.get_vector_store(dataset, hybrid=True)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        chat_engine = index.as_chat_engine(
            memory=memory,
            system_prompt=context_prompt,
            streaming=True,
            **CHAT_ENGINE_PARAMS
        )
        
        return chat_engine
    except Exception as e:
        logger.exception(f"Error creating chat engine for profile {chat_profile}: {e}")
        return None

def setup_runnable():
    """Set up the chat engine runnable in the user session."""
    try:
        chat_profile = cl.user_session.get("chat_profile")
        memory = cl.user_session.get("memory")  # type: ChatMemoryBuffer
        
        if not chat_profile:
            logger.error("chat_profile not found in user session.")
            return
        if not memory:
            logger.error("memory not found in user session.")
            return
        
        llm = get_llm_settings(chat_profile)
        Settings.llm = llm  # Set LLM settings
        
        chat_engine = create_chat_engine(chat_profile, memory)
        if chat_engine:
            cl.user_session.set("runnable", chat_engine)
            # logger.info("Runnable set in user session.")
        else:
            logger.warning("Failed to create chat engine.")
    except Exception as e:
        logger.exception("Error setting up runnable: %s", e)

# # Mock test authentication
# @cl.password_auth_callback
# def auth_callback(username: str, password: str):
#     # Fetch the user matching username from your database
#     # and compare the hashed password with the value stored in the database
#     if (username, password) == ("admin", "admin"):
#         return cl.User(
#             identifier="admin", metadata={"role": "ADMIN", "provider": "credentials"}
#         )
#     else:
#         return None

# OAuth callback function
@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    # Log raw user data for debugging purposes
    # logging.info("Raw user data received from Auth0: %s", raw_user_data)
    return default_user


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    if "GG-TH-CSChatBot-Admin" not in current_user.metadata["groups"]:
        return [
            cl.ChatProfile(
                name="Lotus's AI",
                markdown_description="Lotus's AI for Customer Service",
                # icon="/public/favicon.png",
            )
        ]
    return [
        cl.ChatProfile(
            name="Lotus's AI",
            markdown_description="Lotus's AI for Customer Service",
            # icon="/public/favicon.png",
        ),
        cl.ChatProfile(
            name="Qwen-2.5-32b",
            markdown_description="Lotus's AI for Customer Service powered by Groq",
            # icon="/public/groq_icon.png",
        ),
    ]


# Function that sets four starters for welcome screen
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="โลตัสคอยน์ คืออะไร?",
            message="โลตัสคอยน์ คืออะไร?",
            icon="/public/star.svg",
            ),
        cl.Starter(
            label="My Lotus's คืออะไร?",
            message="My Lotus's คืออะไร?",
            icon="/public/star.svg",
            ),
        cl.Starter(
            label="บัตร Lotus Gift card มีวันหมดอายุหรือไม่?",
            message="บัตร Lotus Gift card มีวันหมดอายุหรือไม่?",
            icon="/public/star.svg",
            ),
        cl.Starter(
            label="ต้องการขอใบกำกับภาษีเมื่อซื้อสินค้าที่สาขา?",
            message="การขอใบกำกับภาษีเมื่อซื้อสินค้าที่สาขา ต้องทำยังไงบ้าง?",
            icon="/public/star.svg",
            )
        ]


@cl.on_chat_start
async def on_chat_start():
    # Access the thread_id from the session context
    thread_id = get_current_chainlit_thread_id()
    app_user = cl.user_session.get("user")
    # Log the user's email and chat start
    user_email = app_user.metadata.get("email", "Unknown")
    logger.info(f"User {user_email} has started new chat session!!")

    # app_user.identifier is UserID
    redis_session_id = f"{app_user.identifier}:{thread_id}"

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=TOKEN_LIMIT,
        chat_store=chat_store,
        chat_store_key=redis_session_id,
    )
    
    cl.user_session.set("memory", memory)
    setup_runnable()  # No need to await


# What to do when chat is resumed from chat history
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    
    thread_id = thread.get("id")
    app_user = cl.user_session.get("user")
    redis_session_id = f"{app_user.identifier}:{thread_id}"
    logger.info("Chat rusume for thread: %s", redis_session_id)
    
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=TOKEN_LIMIT,
        chat_store=chat_store,
        chat_store_key=redis_session_id,
    )

    root_messages = [m for m in thread["steps"] if m["parentId"] is None and m.get("output", "").strip()]
    # messages = [m for m in thread["steps"] if m["type"] in ["user_message", "assistant_message"]]
    # for message in messages:
    for message in root_messages:
        if message["type"] == "user_message":
            message = ChatMessage(role="user", content=message["output"])
            memory.put(message)
        else:
            message = ChatMessage(role="assistant", content=message["output"])
            memory.put(message)
    
    cl.user_session.set("memory", memory)

    setup_runnable()


# Handle user prompt and LLM response
@cl.on_message
async def on_message(message: cl.Message):
    
    runnable = cl.user_session.get("runnable")  
    
    response_message = cl.Message(content="", author="AI Assistant")
    
    streaming_response = await cl.make_async(runnable.stream_chat)(message.content)
    
    for token in streaming_response.response_gen:
        await response_message.stream_token(token=token)
        
    await response_message.send()
