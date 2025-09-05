# OneLogin OIDC Integration Setup

## Overview

This document describes the OneLogin OIDC OAuth integration for the Compliance Chatbot application.

## Configuration

### Environment Variables

The following environment variables are required in both `.env.dev` and `.env.prod`:

```bash
# OneLogin OAuth Configuration
OAUTH_ONELOGIN_CLIENT_ID=your_encrypted_client_credentials
OAUTH_ONELOGIN_CLIENT_SECRET=your_encrypted_client_credentials  
OAUTH_ONELOGIN_DOMAIN=cpxis.global.lotuss.org
OAUTH_ONELOGIN_REDIRECT_URI=http://localhost:8010/compliance/chat/auth/oauth/onelogin/callback  # Dev
# OAUTH_ONELOGIN_REDIRECT_URI=https://cpxis.global.lotuss.org/compliance/chat/auth/oauth/onelogin/callback  # Prod
```

### OneLogin Application Configuration

In your OneLogin application, configure the following:

1. **Redirect URIs**:
   - Development: `http://localhost:8010/compliance/chat/auth/oauth/onelogin/callback`
   - Production: `https://cpxis.global.lotuss.org/compliance/chat/auth/oauth/onelogin/callback`

2. **Scopes**: `openid groups profile params`

3. **Response Type**: `code` (Authorization Code flow)

## File Structure

```
chainlit_app/
├── auth/
│   ├── __init__.py
│   ├── onelogin_oauth_provider.py    # OneLogin OAuth provider implementation
│   └── inject_custom_auth.py         # Authentication helper functions
├── app.py                            # Main application with OAuth integration
├── requirements.txt                  # Updated with httpx dependency
└── test_onelogin_config.py          # Configuration test script
```

## Authentication Flow

1. **User Access**: User visits the chatbot URL
2. **OAuth Redirect**: Chainlit redirects to OneLogin authorization endpoint
3. **User Login**: User enters company credentials in OneLogin
4. **Authorization Code**: OneLogin redirects back with authorization code  
5. **Token Exchange**: App exchanges code for access token
6. **User Info**: App fetches user profile and group information
7. **Session Creation**: Chainlit creates user session with company identity

## User Data Extracted

From OneLogin, the following user data is extracted:

- **Employee ID**: Used as unique identifier
- **Display Name**: Full name from OneLogin
- **Email**: Company email address
- **Groups**: User groups for access control
- **Profile Picture**: User avatar (if available)

## Group-Based Access Control

The application can implement role-based access based on user groups:

```python
@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    user_groups = current_user.metadata.get("groups", [])
    
    # Check if user has admin privileges
    if "ChatBot-Admin" in user_groups:
        # Return admin chat profiles
        return admin_profiles
    else:
        # Return standard user profiles  
        return user_profiles
```

## Testing

### Configuration Test

Run the configuration test to verify setup:

```bash
python test_onelogin_config.py
```

### Manual Testing

1. **Development**:
   ```bash
   chainlit run app.py -h --root-path /compliance/chat --port 8010
   ```

2. **Production**: Deploy with appropriate environment variables

### Troubleshooting

1. **Missing Environment Variables**: 
   - Check `.env.dev` and `.env.prod` files
   - Verify all required OAuth variables are set

2. **Redirect URI Mismatch**:
   - Ensure OneLogin app has correct redirect URIs configured
   - Check dev vs prod URL configurations

3. **Authentication Failures**:
   - Check OneLogin application logs
   - Verify client credentials are correct
   - Ensure user has proper permissions

## Security Notes

- Client credentials are encrypted in environment variables
- OAuth flow uses secure authorization code grant
- User sessions are managed by Chainlit framework
- Group-based access control prevents unauthorized access

## Migration from Password Auth

The old password authentication has been replaced with OneLogin OAuth:

- ❌ **Old**: Username/password authentication
- ✅ **New**: Company SSO via OneLogin OIDC
- ✅ **Benefits**: 
  - Single sign-on with company credentials
  - Automatic user provisioning
  - Group-based access control
  - Enhanced security and audit trails