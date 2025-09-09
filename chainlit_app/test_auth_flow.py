#!/usr/bin/env python3
"""
Test OneLogin authentication flow with test server
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

async def test_auth_flow():
    """Test the complete authentication flow"""
    
    client_id = os.environ.get("OAUTH_ONELOGIN_CLIENT_ID")
    domain = os.environ.get("OAUTH_ONELOGIN_DOMAIN")
    redirect_uri = os.environ.get("OAUTH_ONELOGIN_REDIRECT_URI")
    
    print("=" * 60)
    print("Testing OneLogin Authentication Flow")
    print("=" * 60)
    print(f"\nTest Server Configuration:")
    print(f"Client ID: {client_id}")
    print(f"Domain: {domain}")
    print(f"Redirect URI: {redirect_uri}")
    
    # Test 1: Check OIDC configuration
    print(f"\n1. Testing OIDC Configuration:")
    well_known_url = f"https://{domain}/oidc/2/.well-known/openid-configuration"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(well_known_url)
            if response.status_code == 200:
                config = response.json()
                print(f"✅ OIDC Config accessible")
                print(f"   Authorization endpoint: {config.get('authorization_endpoint')}")
                print(f"   Token endpoint: {config.get('token_endpoint')}")
                print(f"   UserInfo endpoint: {config.get('userinfo_endpoint')}")
            else:
                print(f"❌ OIDC Config failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ OIDC Config error: {e}")
            return False
    
    # Test 2: Test authorization endpoint
    print(f"\n2. Testing Authorization Endpoint:")
    auth_url = f"https://{domain}/oidc/2/auth"
    
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid groups profile params",
        "state": "test_state_123",
        "nonce": "test_nonce_456"
    }
    
    async with httpx.AsyncClient(follow_redirects=False) as client:
        try:
            response = await client.get(auth_url, params=params)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 302 or response.status_code == 303:
                location = response.headers.get("location", "")
                if "error" in location:
                    # Parse error from location
                    parsed = urllib.parse.urlparse(location)
                    error_params = urllib.parse.parse_qs(parsed.query)
                    error = error_params.get("error", [""])[0]
                    error_desc = error_params.get("error_description", [""])[0]
                    print(f"❌ OAuth Error: {error}")
                    print(f"   Description: {error_desc}")
                    return False
                else:
                    print(f"✅ Authorization endpoint working")
                    print(f"   Redirects to: {location[:100]}...")
            else:
                print(f"⚠️  Unexpected status: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ Authorization error: {e}")
            return False
    
    # Test 3: Check if redirect URI needs to be registered
    print(f"\n3. Testing Redirect URI Registration:")
    
    # Try different redirect URI formats to see which one works
    test_uris = [
        redirect_uri,
        "https://localhost:8010/compliance/chat/auth/oauth/onelogin/callback",
        f"https://{domain}/oidc/2/auth",  # Sometimes they expect the auth endpoint itself
    ]
    
    for test_uri in test_uris:
        test_params = params.copy()
        test_params["redirect_uri"] = test_uri
        
        async with httpx.AsyncClient(follow_redirects=False) as client:
            try:
                response = await client.get(auth_url, params=test_params)
                if response.status_code in [302, 303]:
                    location = response.headers.get("location", "")
                    if "error" not in location:
                        print(f"✅ Working redirect URI: {test_uri}")
                        break
                    else:
                        print(f"❌ Failed redirect URI: {test_uri}")
                        # Parse error
                        parsed = urllib.parse.urlparse(location)
                        error_params = urllib.parse.parse_qs(parsed.query)
                        error = error_params.get("error", [""])[0]
                        print(f"   Error: {error}")
            except Exception as e:
                print(f"❌ Error testing {test_uri}: {e}")
    
    print(f"\n4. Manual Test URL:")
    manual_url = f"{auth_url}?{urllib.parse.urlencode(params)}"
    print(f"Copy and test this URL in your browser:")
    print(f"{manual_url}")
    print(f"\nIf you get a login page, the test server is working!")
    print(f"If you get an error, check the OneLogin application configuration.")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_auth_flow())