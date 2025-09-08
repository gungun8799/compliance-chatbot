#!/usr/bin/env python3
"""
Test script to verify the new OneLogin OAuth configuration
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent.parent / '.env.dev'
load_dotenv(env_file)

def test_onelogin_config():
    """Test OneLogin OAuth configuration"""
    print("=" * 60)
    print("Testing OneLogin OAuth Configuration")
    print("=" * 60)
    
    # Check required environment variables
    required_vars = [
        "OAUTH_ONELOGIN_CLIENT_ID",
        "OAUTH_ONELOGIN_CLIENT_SECRET", 
        "OAUTH_ONELOGIN_DOMAIN",
        "OAUTH_ONELOGIN_REDIRECT_URI"
    ]
    
    all_present = True
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            if "SECRET" in var:
                display_value = f"{value[:10]}..." if len(value) > 10 else value
            else:
                display_value = value
            print(f"✓ {var}: {display_value}")
        else:
            print(f"✗ {var}: NOT SET")
            all_present = False
    
    if not all_present:
        print("\n❌ Some required environment variables are missing!")
        return False
    
    # Display the OAuth URLs that will be used
    domain = os.environ.get("OAUTH_ONELOGIN_DOMAIN", "").rstrip('/')
    print("\n" + "=" * 60)
    print("OAuth URLs (constructed from domain):")
    print("=" * 60)
    print(f"Authorize URL: https://{domain}/oidc/2/auth")
    print(f"Token URL: https://{domain}/oidc/2/token")
    print(f"UserInfo URL: https://{domain}/oidc/2/me")
    print(f"Issuer URL: https://{domain}/oidc/2/.well-known/openid-configuration")
    
    # Display redirect URIs
    print("\n" + "=" * 60)
    print("Redirect URI Configuration:")
    print("=" * 60)
    redirect_uri = os.environ.get("OAUTH_ONELOGIN_REDIRECT_URI")
    print(f"Development: {redirect_uri}")
    
    # Check if we can import the provider
    print("\n" + "=" * 60)
    print("Testing Provider Import:")
    print("=" * 60)
    try:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        from auth.onelogin_oauth_provider import OneLoginOAuthProvider
        
        # Try to instantiate the provider
        provider = OneLoginOAuthProvider()
        print("✓ Successfully imported and instantiated OneLoginOAuthProvider")
        
        # Display provider configuration
        print(f"\nProvider Configuration:")
        print(f"  Client ID: {provider.client_id[:20]}...")
        print(f"  Domain: {provider.domain}")
        print(f"  Authorize URL: {provider.authorize_url}")
        print(f"  Token URL: {provider.token_url}")
        print(f"  UserInfo URL: {provider.userinfo_url}")
        
    except ImportError as e:
        print(f"✗ Failed to import provider: {e}")
        return False
    except Exception as e:
        print(f"✗ Failed to instantiate provider: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ OneLogin OAuth configuration test completed successfully!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Ensure OneLogin application is configured with the redirect URI")
    print("2. Test the OAuth flow by running the Chainlit app")
    print("3. Monitor logs for any authentication issues")
    
    return True

if __name__ == "__main__":
    success = test_onelogin_config()
    sys.exit(0 if success else 1)