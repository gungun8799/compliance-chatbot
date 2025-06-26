import logging
import httpx
from datetime import datetime
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
import llama_index.storage.chat_store.redis.base as base  # Import the specific module to patch
from llama_index.core.llms import ChatMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Define the patched synchronous method
def patched_call_api(self, texts: list) -> list:
    headers = {"Content-Type": "application/json"}
    if self.auth_token is not None:
        if callable(self.auth_token):
            headers["Authorization"] = self.auth_token(self.base_url)
        else:
            headers["Authorization"] = self.auth_token
    json_data = {"inputs": texts, "truncate": self.truncate_text}

    # Disable SSL verification here
    with httpx.Client(verify=False) as client:
        response = client.post(
            f"{self.base_url}/embed",
            headers=headers,
            json=json_data,
            timeout=self.timeout
        )
    return response.json()

# Define the patched asynchronous method
async def patched_acall_api(self, texts: list) -> list:
    headers = {"Content-Type": "application/json"}
    if self.auth_token is not None:
        if callable(self.auth_token):
            headers["Authorization"] = self.auth_token(self.base_url)
        else:
            headers["Authorization"] = self.auth_token
    json_data = {"inputs": texts, "truncate": self.truncate_text}

    # Disable SSL verification here
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.post(
            f"{self.base_url}/embed",
            headers=headers,
            json=json_data,
            timeout=self.timeout
        )
    return response.json()

# Define the new _message_to_dict function with created_time
def _message_to_dict_with_timestamp(message: ChatMessage) -> dict:
    return {**message.dict(), "created_time": datetime.now().isoformat()}

# Apply the monkey-patch
def apply_patch():
    # Patch _call_api and _acall_api in TextEmbeddingsInference
    TextEmbeddingsInference._call_api = patched_call_api
    logger.info("Patched TextEmbeddingsInference._call_api with custom synchronous API handling")

    TextEmbeddingsInference._acall_api = patched_acall_api
    logger.info("Patched TextEmbeddingsInference._acall_api with custom asynchronous API handling")
    
    # Replace _message_to_dict in base.py with the new function
    # base._message_to_dict = _message_to_dict_with_timestamp
    # logger.info("Patched _message_to_dict in base.py with custom timestamp handling")