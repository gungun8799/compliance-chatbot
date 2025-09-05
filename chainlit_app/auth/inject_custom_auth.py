import os
import secrets
import string
import logging
from chainlit.oauth_providers import providers
# from onelogin_oauth_provider import OneLoginOAuthProvider  

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chars = string.ascii_letters + string.digits + "$%*,-./:=>?@^_~"

def random_secret(length: int = 64):
    return "".join(secrets.choice(chars) for _ in range(length))

def onelogin_oauth_enabled():
    required_vars = [
        'OAUTH_ONELOGIN_CLIENT_ID',
        'OAUTH_ONELOGIN_CLIENT_SECRET',
        'OAUTH_ONELOGIN_DOMAIN',
        'OAUTH_ONELOGIN_REDIRECT_URI'
    ]
    missing_vars = [var for var in required_vars if os.environ.get(var) is None]
    if missing_vars:
        logger.warning(f"OneLogin OAuth not configured. Missing variables: {', '.join(missing_vars)}. Skipping...")
        return False
    logger.info("OneLogin OAuth configured.")
    return True

def provider_id_in_instance_list(provider_id: str):
    if providers is None:
        logger.error("No providers found")
        return False
    if not any(provider.id == provider_id for provider in providers):
        logger.warning(f"Provider {provider_id} not found")
        return False
    logger.info(f"Provider {provider_id} found")
    return True

def add_custom_oauth_provider(provider_id: str, custom_provider_instance):
    if onelogin_oauth_enabled() and not provider_id_in_instance_list(provider_id):
        providers.append(custom_provider_instance)
        logger.info(f"Added provider: {provider_id}")
    else:
        logger.info(f"Custom OAuth is not enabled or provider {provider_id} already exists")

# Example usage
# if __name__ == "__main__":
#     # Instantiate OneLoginOAuthProvider and add it to providers
#     onelogin_provider = OneLoginOAuthProvider()
#     add_custom_oauth_provider(onelogin_provider.id, onelogin_provider)