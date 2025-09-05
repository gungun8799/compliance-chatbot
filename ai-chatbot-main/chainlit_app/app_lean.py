from llama_index.llms.groq import Groq
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from chainlit.types import ThreadDict
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
from dotenv import load_dotenv
import chainlit as cl
import os

# Load variables from the .env file
load_dotenv()
# Access the variables
REDIS_CHATSTORE_URI = os.getenv("REDIS_CHATSTORE_URI")
REDIS_CHATSTORE_PASSWORD = os.getenv("REDIS_CHATSTORE_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID")
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

def get_current_chainlit_thread_id() -> str:
    return cl.context.session.thread_id

def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ChatMemoryBuffer
    
    llm = Groq(model=GROQ_MODEL_ID, api_key=GROQ_API_KEY)
    
    system_prompt = "You are an intelligent, helpful, and patient AI assistant. Your role is to provide clear, insightful, and friendly responses to guide users effectively. Focus on understanding user needs, adapting your answers to their level of expertise, and delivering information in an accurate and concise manner."
    
    chat_engine = SimpleChatEngine.from_defaults(llm=llm, system_prompt=system_prompt, memory=memory)

    cl.user_session.set("runnable", chat_engine)


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


#Function that sets four starters for welcome screen
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


#Set app environment before start
@cl.on_chat_start
async def on_chat_start():
    # Access the thread_id from the session context
    thread_id =  get_current_chainlit_thread_id()
    # print("Current thread_id is", thread_id, "(on_chat_start)")
    app_user = cl.user_session.get("user")
    redis_session_id = f"{app_user.identifier}:{thread_id}"

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=2000,
        chat_store=chat_store,
        chat_store_key=redis_session_id,
    )
    
    cl.user_session.set("memory", memory)
    setup_runnable()


#What to do when chat is resumed from chat history
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    
    thread_id = thread.get("id")
    app_user = cl.user_session.get("user")
    redis_session_id = f"{app_user.identifier}:{thread_id}"
    
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=2000,
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


#Handle user prompt and LLM response
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  
    
    response_message = cl.Message(content="", author="AI Assistant")
    
    response = await cl.make_async(runnable.chat)(message.content)
    
    for token in response.response:
        await response_message.stream_token(token=token)
        
    await response_message.send()