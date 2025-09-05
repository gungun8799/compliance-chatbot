import os
import httpx
from base64 import b64encode
from fastapi import HTTPException
from chainlit.user import User
from chainlit.oauth_providers import OAuthProvider
import secrets

class OneLoginOAuthProvider(OAuthProvider):
    id = "onelogin"
    env = [
        "OAUTH_ONELOGIN_CLIENT_ID",
        "OAUTH_ONELOGIN_CLIENT_SECRET",
        "OAUTH_ONELOGIN_DOMAIN",
        "OAUTH_ONELOGIN_REDIRECT_URI"
    ]

    def __init__(self):
        self.client_id = os.environ.get("OAUTH_ONELOGIN_CLIENT_ID")
        self.client_secret = os.environ.get("OAUTH_ONELOGIN_CLIENT_SECRET")
        self.domain = os.environ.get("OAUTH_ONELOGIN_DOMAIN", "").rstrip('/')
        self.redirect_uri = os.environ.get("OAUTH_ONELOGIN_REDIRECT_URI")

        # Validate environment variables
        missing_vars = [var for var in self.env if not os.environ.get(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        self.authorize_url = f"https://{self.domain}/oidc/2/auth"
        self.token_url = f"https://{self.domain}/oidc/2/token"
        self.userinfo_url = f"https://{self.domain}/oidc/2/me"
        
    @property
    def authorize_params(self):
        """Generates and returns authorization parameters with a unique nonce."""
        nonce = secrets.token_urlsafe(16)  # Generate a secure random nonce
        authorize_params = {
            "response_type": "code",
            "scope": "openid groups profile params",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "nonce": nonce,
        }

        prompt = self.get_prompt()
        if prompt:
            authorize_params["prompt"] = prompt

        return authorize_params

    async def get_token(self, code: str, redirect_uri: str):
        # Prepare headers
        basic_auth = b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {basic_auth}",
        }
        # print(f"URL from Request: {redirect_uri}")
        # Prepare payload with data-urlencode style
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }

        async with httpx.AsyncClient() as client:
            try:
                # print("Request Payload:", payload)
                response = await client.post(self.token_url, headers=headers, data=payload)
                response.raise_for_status()
                json_content = response.json()
                token = json_content.get("access_token")
                if not token:
                    raise HTTPException(
                        status_code=400, detail="Access token not found in the response"
                    )
                return token
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Failed to obtain access token: {e.response.text}"
                )

    async def get_user_info(self, token: str):
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.userinfo_url, headers=headers)
                response.raise_for_status()
                onelogin_user = response.json()
                # Safely extract EmployeeID and other fields
                employee_id = onelogin_user.get("params", {}).get("EmployeeID", "unknown")
                user_groups = onelogin_user.get("groups", [])
                name = onelogin_user.get("name", "unknown")
                email = onelogin_user.get("email", "unknown")
                picture = onelogin_user.get("picture", "")
                
                # Create a `User` object with the extracted data
                user = User(
                    identifier=employee_id,  # Unique employee ID
                    display_name=name,       # Full name of the user
                    metadata={
                        "image": picture,    # User's profile picture
                        "email": email,      # User's email address
                        "provider": "onelogin",  # OAuth provider
                        # "role": "ADMIN",     # Role assigned to the user (customizable)
                        "groups": user_groups
                    },
                )
                
                return onelogin_user, user
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Failed to retrieve user info: {e.response.text}"
                )