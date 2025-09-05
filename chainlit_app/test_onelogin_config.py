#!/usr/bin/env python3
"""
Test script to validate OneLogin OAuth configuration
"""
import os
import sys
from pathlib import Path

# Add parent directory to sys.path to import from .env files
sys.path.append(str(Path(__file__).parent.parent))

# Load environment
from dotenv import load_dotenv

def test_onelogin_config(env_mode="dev"):
    """Test OneLogin configuration for specified environment"""
    print(f"🧪 Testing OneLogin configuration for {env_mode.upper()} environment")
    
    # Load appropriate environment file
    env_file = Path(__file__).parent.parent / f".env.{env_mode}"
    if not env_file.exists():
        print(f"❌ Environment file not found: {env_file}")
        return False
    
    load_dotenv(dotenv_path=env_file)
    
    # Check required environment variables
    required_vars = [
        "OAUTH_ONELOGIN_CLIENT_ID",
        "OAUTH_ONELOGIN_CLIENT_SECRET", 
        "OAUTH_ONELOGIN_DOMAIN",
        "OAUTH_ONELOGIN_REDIRECT_URI"
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            missing_vars.append(var)
        else:
            print(f"✅ {var}: {'*' * min(len(value), 10)}... (configured)")
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    # Test OAuth provider initialization
    try:
        from auth.onelogin_oauth_provider import OneLoginOAuthProvider
        from auth.inject_custom_auth import add_custom_oauth_provider, onelogin_oauth_enabled
        
        if not onelogin_oauth_enabled():
            print("❌ OneLogin OAuth not enabled")
            return False
            
        print("✅ OneLogin OAuth configuration check passed")
        
        # Create provider instance
        provider = OneLoginOAuthProvider()
        print(f"✅ OneLogin provider created successfully")
        print(f"   Domain: {provider.domain}")
        print(f"   Authorize URL: {provider.authorize_url}")
        print(f"   Token URL: {provider.token_url}")
        print(f"   User Info URL: {provider.userinfo_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating OneLogin provider: {e}")
        return False

def main():
    """Main test function"""
    print("🔐 OneLogin OAuth Configuration Test")
    print("=" * 50)
    
    # Test dev environment
    dev_success = test_onelogin_config("dev")
    print()
    
    # Test prod environment  
    prod_success = test_onelogin_config("prod")
    print()
    
    if dev_success and prod_success:
        print("🎉 All OneLogin configuration tests passed!")
        print("\n📋 Next steps:")
        print("1. Ensure your OneLogin application is configured with the redirect URIs")
        print("2. Run the chatbot with: chainlit run app.py -h --root-path /compliance/chat")
        print("3. Test login with your company credentials")
        return True
    else:
        print("❌ Some configuration tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)