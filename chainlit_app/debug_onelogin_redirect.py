#!/usr/bin/env python3
"""
Debug script to understand OneLogin redirect URI requirements
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import urllib.parse

def main():
    # Load environment
    env_file = Path(__file__).parent.parent / ".env.dev"
    load_dotenv(env_file)
    
    print("=" * 60)
    print("OneLogin Redirect URI Debug Information")
    print("=" * 60)
    
    # Current configuration
    client_id = os.environ.get("OAUTH_ONELOGIN_CLIENT_ID")
    domain = os.environ.get("OAUTH_ONELOGIN_DOMAIN")
    redirect_uri = os.environ.get("OAUTH_ONELOGIN_REDIRECT_URI")
    
    print("\n📋 Current Configuration:")
    print(f"Client ID: {client_id}")
    print(f"Domain: {domain}")
    print(f"Redirect URI: {redirect_uri}")
    
    print("\n🔍 Possible Redirect URI Formats to Register in OneLogin:")
    print("-" * 50)
    
    # Generate possible redirect URI variations
    redirect_variations = [
        # HTTP variations (for local development)
        "http://localhost:8010/compliance/chat/auth/oauth/onelogin/callback",
        "http://localhost:8010/auth/oauth/onelogin/callback",
        "http://localhost/compliance/chat/auth/oauth/onelogin/callback",
        
        # HTTPS variations (for local development with SSL)
        "https://localhost:8010/compliance/chat/auth/oauth/onelogin/callback",
        "https://localhost:8010/auth/oauth/onelogin/callback",
        "https://localhost/compliance/chat/auth/oauth/onelogin/callback",
        
        # Production variations
        "https://cpxis.global.lotuss.org/compliance/chat/auth/oauth/onelogin/callback",
        "https://cpxis.global.lotuss.org/auth/oauth/onelogin/callback",
    ]
    
    print("\n📝 Copy and add ALL of these to your OneLogin app configuration:")
    for i, uri in enumerate(redirect_variations, 1):
        print(f"{i}. {uri}")
    
    print("\n⚠️  IMPORTANT:")
    print("1. In OneLogin admin, go to your application settings")
    print("2. Find the 'Redirect URIs' or 'Callback URLs' section")
    print("3. Add ALL the URIs listed above")
    print("4. Save the changes")
    print("\n5. The redirect URI in the OAuth request MUST match")
    print("   EXACTLY one of the registered URIs (including protocol and port)")
    
    print("\n🔗 Test URLs for different scenarios:")
    print("-" * 50)
    
    # Generate test URLs
    auth_endpoint = f"https://{domain}/oidc/2/auth"
    
    for scenario, redirect in [
        ("Local HTTP", "http://localhost:8010/compliance/chat/auth/oauth/onelogin/callback"),
        ("Local HTTPS", "https://localhost:8010/compliance/chat/auth/oauth/onelogin/callback"),
        ("Production", "https://cpxis.global.lotuss.org/compliance/chat/auth/oauth/onelogin/callback")
    ]:
        params = {
            "client_id": client_id,
            "redirect_uri": redirect,
            "response_type": "code",
            "scope": "openid groups profile params",
            "state": "test_state",
            "nonce": "test_nonce"
        }
        test_url = f"{auth_endpoint}?{urllib.parse.urlencode(params)}"
        print(f"\n{scenario}:")
        print(f"  Redirect URI: {redirect}")
        print(f"  Test URL: {test_url[:100]}...")
    
    print("\n" + "=" * 60)
    print("📌 Next Steps:")
    print("1. Add ALL redirect URIs to OneLogin")
    print("2. Wait a few minutes for changes to propagate")
    print("3. Try accessing: https://localhost:8010/compliance/chat")
    print("=" * 60)

if __name__ == "__main__":
    main()