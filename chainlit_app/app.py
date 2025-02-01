# Run application locally using this command: chainlit run app.py -h --root-path /chatbot/v1
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.groq import Groq
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from functions.qdrant_vectordb import QdrantManager
from chainlit.types import ThreadDict
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
from typing import Dict, Optional
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT_STANDARD, SYSTEM_PROMPT_DEEPTHINK
import chainlit as cl
import os
import time
import logging
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load variables from the .env file
load_dotenv()
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
Settings.context_window = 6000

# Set up embedding model
Settings.embed_model = CohereEmbedding(
    api_key=os.getenv("COHEAR_API_KEY"),
    model_name=os.getenv("COHEAR_MODEL_ID"),
    input_type="search_document",
    embedding_type="float",
)

def get_current_chainlit_thread_id() -> str:
    return cl.context.session.thread_id

# Constants and configurations
DATASET_MAPPING = {
    "Standard": QDANT_COLLENCTION_NAME,
    "Deepthink": QDANT_COLLENCTION_NAME,
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
}

def load_context_prompt(chat_profile: str) -> str:
    """Load the context prompt for the given chat profile."""
    return CHAT_PROFILES.get(chat_profile, {}).get("context_prompt", "")

def get_llm_settings(chat_profile: str):
    """Retrieve and configure LLM settings based on the chat profile."""
    settings = CHAT_PROFILES.get(chat_profile, {}).get("llm_settings")
    if not settings:
        raise ValueError(f"No LLM settings found for profile: {chat_profile}")
    
    if chat_profile == "Standard":
        return Groq(
            model=settings["model"],
            api_key=settings["api_key"],
            is_chat_model=settings["is_chat_model"],
            is_function_calling_model=settings["is_function_calling_model"],
            temperature=settings["temperature"],
        )
    elif chat_profile == "Deepthink":
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
    """Create a chat engine based on the specified chat profile."""
    # logger.info("Creating chat engine for profile: %s", chat_profile)
    
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
        memory = cl.user_session.get("memory")  
        
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

# Mock test authentication
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "ADMIN", "email": "chatbot_admin@gmail.com","provider": "credentials"}
        )
    else:
        return None

@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    return [
        cl.ChatProfile(
            name="Standard",
            markdown_description="Powered by llama-3.3-70b-versatile (Groq) model.",
            icon="/public/meta-color.png",
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
            label="What is Walmart+ Travel?",
            message="What is Walmart+ Travel?",
            icon="/public/search.svg",
            ),
        cl.Starter(
            label="Does Walmart Cash expire?",
            message="Does Walmart Cash expire?",
            icon="/public/search.svg",
            ),
        cl.Starter(
            label="What is Pawp?",
            message="What is Pawp?",
            icon="/public/search.svg",
            ),
        cl.Starter(
            label="What’s Walmart+ InHome?",
            message="What’s Walmart+ InHome?",
            icon="/public/search.svg",
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
    # logger.info("Chat rusume for thread: %s", redis_session_id)
    
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
    
@cl.on_message
async def on_message(message: cl.Message):
    start = time.time()
    
    runnable = cl.user_session.get("runnable")
    chat_profile = cl.user_session.get("chat_profile")  # Get current chat profile
    
    streaming_response = await cl.make_async(runnable.stream_chat)(message.content)

    if chat_profile == "Deepthink":
        # Use Thinking step for Deepthink profile
        thinking = False
        async with cl.Step(name="Thinking") as thinking_step:
            final_answer = cl.Message(content="", author="Customer Service Agent")

            for token in streaming_response.response_gen:
                if token == "<think>":
                    thinking = True
                    continue

                if token == "</think>":
                    thinking = False
                    thought_for = round(time.time() - start)
                    thinking_step.name = f"Thought for {thought_for}s"
                    await thinking_step.update()
                    continue

                if thinking:
                    await thinking_step.stream_token(token)
                else:
                    await final_answer.stream_token(token)

            await final_answer.send()
    else:
        # Direct streaming for Standard profile
        final_answer = cl.Message(content="", author="Customer Service Agent")
        
        for token in streaming_response.response_gen:
            # Skip any potential think tags for non-Deepthink profiles
            if token in ["<think>", "</think>"]:
                continue
            await final_answer.stream_token(token)
        
        await final_answer.send()
