#!/usr/bin/env python3
"""
Test OneLogin authentication with different redirect URI formats
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import urllib.parse
import httpx
import asyncio

# Load environment
env_file = Path(__file__).parent.parent / ".env.dev"
load_dotenv(env_file)

async def test_auth_with_redirect(redirect_uri):
    """Test authorization with a specific redirect URI"""
    client_id = os.environ.get("OAUTH_ONELOGIN_CLIENT_ID")
    domain = os.environ.get("OAUTH_ONELOGIN_DOMAIN")
    
    auth_endpoint = f"https://{domain}/oidc/2/auth"
    
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid groups profile params",
        "state": "test123",
        "nonce": "nonce123"
    }
    
    async with httpx.AsyncClient(follow_redirects=False) as client:
        try:
            response = await client.get(auth_endpoint, params=params)
            return response.status_code, response.headers.get("location", "")
        except Exception as e:
            return None, str(e)

async def main():
    print("=" * 60)
    print("Testing OneLogin Authentication with Different Redirect URIs")
    print("=" * 60)
    
    client_id = os.environ.get("OAUTH_ONELOGIN_CLIENT_ID")
    print(f"\nClient ID: {client_id}")
    print(f"Domain: {os.environ.get('OAUTH_ONELOGIN_DOMAIN')}")
    
    # Test different redirect URI variations
    redirect_uris = [
        "http://localhost:8010/compliance/chat/auth/oauth/onelogin/callback",
        "https://localhost:8010/compliance/chat/auth/oauth/onelogin/callback",
        "http://localhost/compliance/chat/auth/oauth/onelogin/callback",
        "https://localhost/compliance/chat/auth/oauth/onelogin/callback",
        "http://127.0.0.1:8010/compliance/chat/auth/oauth/onelogin/callback",
        "https://cpxis.global.lotuss.org/compliance/chat/auth/oauth/onelogin/callback",
    ]
    
    print("\n" + "=" * 60)
    print("Testing Redirect URIs:")
    print("=" * 60)
    
    for uri in redirect_uris:
        print(f"\nTesting: {uri}")
        status, location = await test_auth_with_redirect(uri)
        
        if status == 302 or status == 303:
            # Check if it's redirecting to login page (success) or error page
            if "error" in location:
                # Parse error from location
                parsed = urllib.parse.urlparse(location)
                params = urllib.parse.parse_qs(parsed.query)
                error = params.get("error", [""])[0]
                error_desc = params.get("error_description", [""])[0]
                print(f"  ❌ Error: {error}")
                print(f"     {error_desc}")
            else:
                print(f"  ✅ Success! Redirecting to login page")
                print(f"     Location: {location[:100]}...")
        elif status == 400:
            print(f"  ❌ Bad Request (400) - Invalid parameters")
        else:
            print(f"  ⚠️  Status: {status}")
    
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    print("✅ URIs that redirect to login are correctly registered in OneLogin")
    print("❌ URIs with 'redirect_uri_mismatch' need to be added to OneLogin")
    print("\nIn OneLogin Admin:")
    print("1. Go to your application settings")
    print("2. Find 'Redirect URIs' or 'Callback URLs'")
    print("3. Add the URIs that failed above")
    print("4. Save and wait a few minutes for changes to propagate")

if __name__ == "__main__":
    asyncio.run(main())