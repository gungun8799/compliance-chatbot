#!/usr/bin/env python3
"""Test script to check the actual OAuth URL being generated"""

import os
import sys
from pathlib import Path

# Load environment
sys.path.append(str(Path(__file__).parent.parent))
from dotenv import load_dotenv

env_file = Path(__file__).parent.parent / ".env.dev"
load_dotenv(dotenv_path=env_file)

# Import and test the provider
from auth.onelogin_oauth_provider import OneLoginOAuthProvider
from auth.inject_custom_auth import add_custom_oauth_provider
from chainlit.oauth_providers import providers

print("=" * 50)
print("OneLogin OAuth URL Test")
print("=" * 50)

# Check environment
domain = os.environ.get("OAUTH_ONELOGIN_DOMAIN")
client_id = os.environ.get("OAUTH_ONELOGIN_CLIENT_ID")
redirect_uri = os.environ.get("OAUTH_ONELOGIN_REDIRECT_URI")

print(f"Environment Domain: {domain}")
print(f"Client ID: {client_id[:20]}...")
print(f"Redirect URI: {redirect_uri}")
print()

# Create provider
provider = OneLoginOAuthProvider()
print(f"Provider Domain: {provider.domain}")
print(f"Provider Auth URL: {provider.authorize_url}")
print()

# Check authorize params
params = provider.authorize_params
print("Authorize Parameters:")
for key, value in params.items():
    if key == "client_id":
        print(f"  {key}: {value[:20]}...")
    else:
        print(f"  {key}: {value}")
print()

# Build full URL
import urllib.parse
full_url = f"{provider.authorize_url}?{urllib.parse.urlencode(params)}"
print(f"Full OAuth URL that would be generated:")
print(full_url[:200] + "...")
print()

# Check if domain is correct
if "ourlogintest" in full_url:
    print("❌ ERROR: Still using ourlogintest domain!")
elif "ourlogin.lotuss.com" in full_url:
    print("✅ SUCCESS: Using correct ourlogin.lotuss.com domain!")
else:
    print("⚠️ WARNING: Unexpected domain in URL")