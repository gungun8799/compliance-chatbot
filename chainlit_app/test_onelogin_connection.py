#!/usr/bin/env python3
"""
OneLogin Connection Test Script
Tests the OneLogin OIDC endpoints and configuration
"""
import os
import sys
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv
import urllib.parse

async def test_onelogin_endpoints():
    """Test OneLogin OIDC endpoints"""
    
    # Load development environment
    env_file = Path(__file__).parent.parent / ".env.dev"
    load_dotenv(dotenv_path=env_file)
    
    # Get OneLogin configuration
    domain = os.environ.get("OAUTH_ONELOGIN_DOMAIN")
    client_id = os.environ.get("OAUTH_ONELOGIN_CLIENT_ID")
    redirect_uri = os.environ.get("OAUTH_ONELOGIN_REDIRECT_URI")
    
    print("🔐 OneLogin Connection Test")
    print("=" * 50)
    print(f"Domain: {domain}")
    print(f"Client ID: {client_id[:20]}..." if client_id else "Not configured")
    print(f"Redirect URI: {redirect_uri}")
    print()
    
    if not all([domain, client_id, redirect_uri]):
        print("❌ Missing required OneLogin configuration")
        return False
    
    # Test endpoints
    base_url = f"https://{domain}"
    endpoints = {
        "Authorization": f"{base_url}/oidc/2/auth",
        "Token": f"{base_url}/oidc/2/token", 
        "UserInfo": f"{base_url}/oidc/2/me",
        "Well-known": f"{base_url}/.well-known/openid_configuration"
    }
    
    async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
        for name, url in endpoints.items():
            try:
                print(f"🧪 Testing {name}: {url}")
                response = await client.get(url)
                
                if response.status_code == 200:
                    print(f"✅ {name}: OK (200)")
                elif response.status_code == 404:
                    print(f"❌ {name}: Not Found (404) - Endpoint doesn't exist")
                elif response.status_code in [400, 401, 403]:
                    print(f"⚠️ {name}: Client Error ({response.status_code}) - Endpoint exists but requires auth")
                else:
                    print(f"⚠️ {name}: Response ({response.status_code})")
                    
            except httpx.TimeoutException:
                print(f"⏱️ {name}: Timeout - Server not responding")
            except httpx.ConnectError:
                print(f"❌ {name}: Connection Error - Cannot reach server")
            except Exception as e:
                print(f"❌ {name}: Error - {str(e)}")
    
    print()
    
    # Test authorization URL construction
    print("🔗 Testing Authorization URL Construction")
    print("-" * 30)
    
    auth_url = f"{base_url}/oidc/2/auth"
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid groups profile params",
        "state": "test_state_123",
        "nonce": "test_nonce_456"
    }
    
    full_auth_url = f"{auth_url}?" + urllib.parse.urlencode(params)
    print(f"Full Auth URL: {full_auth_url}")
    print()
    
    # Test the auth URL accessibility
    try:
        print("🧪 Testing Authorization URL accessibility...")
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(full_auth_url, follow_redirects=False)
            
            if response.status_code == 200:
                print("✅ Authorization URL accessible - Should show login page")
                return True
            elif response.status_code == 302:
                print(f"✅ Authorization URL redirecting - Location: {response.headers.get('location', 'N/A')}")
                return True
            elif response.status_code == 400:
                print("❌ Bad Request (400) - Check client_id or parameters")
                print("Response:", response.text[:200])
                return False
            elif response.status_code == 404:
                print("❌ Not Found (404) - Authorization endpoint doesn't exist")
                return False
            else:
                print(f"⚠️ Status: {response.status_code}")
                print("Response:", response.text[:200])
                return False
                
    except Exception as e:
        print(f"❌ Error testing authorization URL: {e}")
        return False

async def test_well_known_config():
    """Test OneLogin's well-known configuration"""
    env_file = Path(__file__).parent.parent / ".env.dev"
    load_dotenv(dotenv_path=env_file)
    
    domain = os.environ.get("OAUTH_ONELOGIN_DOMAIN")
    well_known_url = f"https://{domain}/.well-known/openid_configuration"
    
    print("🔍 Testing OpenID Configuration")
    print("-" * 30)
    
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(well_known_url)
            
            if response.status_code == 200:
                config = response.json()
                print("✅ OpenID Configuration found:")
                print(f"   Issuer: {config.get('issuer', 'N/A')}")
                print(f"   Auth Endpoint: {config.get('authorization_endpoint', 'N/A')}")
                print(f"   Token Endpoint: {config.get('token_endpoint', 'N/A')}")
                print(f"   UserInfo Endpoint: {config.get('userinfo_endpoint', 'N/A')}")
                return True
            else:
                print(f"❌ OpenID Configuration not found: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Error fetching OpenID configuration: {e}")
        return False

def provide_recommendations():
    """Provide troubleshooting recommendations"""
    print("\n🛠️ Troubleshooting Recommendations")
    print("=" * 50)
    print("1. If all endpoints return 404:")
    print("   - Verify the OneLogin domain is correct")
    print("   - Check if OIDC is enabled on this OneLogin instance")
    print()
    print("2. If authorization endpoint returns 400:")
    print("   - The client_id may be encrypted and needs decryption")
    print("   - Ask IT team for the actual (decrypted) client credentials")
    print()
    print("3. If connection errors occur:")
    print("   - Check if you're behind a corporate firewall")
    print("   - Verify network connectivity to OneLogin servers")
    print()
    print("4. If endpoints exist but return errors:")
    print("   - The OneLogin application may not be properly configured")
    print("   - Ask IT team to verify the application setup in OneLogin")

async def main():
    print("🚀 Starting OneLogin Connection Test")
    print("=" * 50)
    
    # Test endpoints
    endpoints_ok = await test_onelogin_endpoints()
    print()
    
    # Test well-known configuration
    config_ok = await test_well_known_config()
    print()
    
    # Provide recommendations
    provide_recommendations()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 20)
    if endpoints_ok and config_ok:
        print("✅ OneLogin connection appears to be working!")
        print("   The OAuth flow should work now.")
    else:
        print("❌ OneLogin connection issues detected.")
        print("   Check the recommendations above.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")