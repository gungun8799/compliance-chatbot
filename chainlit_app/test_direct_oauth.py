#!/usr/bin/env python3
"""
Direct test of OneLogin OAuth flow
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import urllib.parse
import webbrowser

# Load environment
env_file = Path(__file__).parent.parent / ".env.dev"
load_dotenv(env_file)

def test_oauth_flow():
    """Generate and test the OAuth URL"""
    
    client_id = os.environ.get("OAUTH_ONELOGIN_CLIENT_ID")
    domain = os.environ.get("OAUTH_ONELOGIN_DOMAIN")
    redirect_uri = os.environ.get("OAUTH_ONELOGIN_REDIRECT_URI")
    
    print("=" * 60)
    print("Direct OneLogin OAuth Flow Test")
    print("=" * 60)
    print(f"\nClient ID: {client_id}")
    print(f"Domain: {domain}")
    print(f"Redirect URI: {redirect_uri}")
    
    # Build the authorization URL
    auth_endpoint = f"https://{domain}/oidc/2/auth"
    
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid groups profile params",
        "state": "test_state_123",
        "nonce": "test_nonce_456"
    }
    
    auth_url = f"{auth_endpoint}?{urllib.parse.urlencode(params)}"
    
    print("\n" + "=" * 60)
    print("Authorization URL:")
    print("=" * 60)
    print(auth_url)
    
    print("\n" + "=" * 60)
    print("Decoded Parameters:")
    print("=" * 60)
    for key, value in params.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("Instructions:")
    print("=" * 60)
    print("1. Copy the Authorization URL above")
    print("2. Paste it in your browser")
    print("3. Check the error message from OneLogin")
    print("\nIf you see 'redirect_uri_mismatch', the exact redirect URI shown above")
    print("needs to be registered in your OneLogin application.")
    
    print("\n" + "=" * 60)
    print("Alternative Redirect URIs to Try:")
    print("=" * 60)
    
    alternatives = [
        "http://localhost:8010/compliance/chat/auth/oauth/onelogin/callback",
        "https://localhost:8010/compliance/chat/auth/oauth/onelogin/callback",
        "http://localhost:8010/auth/oauth/onelogin/callback",
        "https://localhost:8010/auth/oauth/onelogin/callback",
    ]
    
    for alt in alternatives:
        params_alt = params.copy()
        params_alt["redirect_uri"] = alt
        alt_url = f"{auth_endpoint}?{urllib.parse.urlencode(params_alt)}"
        print(f"\nWith redirect_uri: {alt}")
        print(f"URL: {alt_url[:100]}...")
    
    # Open in browser?
    response = input("\nDo you want to open the first URL in your browser? (y/n): ")
    if response.lower() == 'y':
        webbrowser.open(auth_url)
        print("URL opened in browser!")

if __name__ == "__main__":
    test_oauth_flow()