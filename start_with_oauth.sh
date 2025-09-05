#!/bin/bash

# Startup script for Compliance Chatbot with OneLogin OAuth
# This script properly loads environment variables before starting the application

echo "🚀 Starting Compliance Chatbot with OneLogin OAuth Integration"
echo "=============================================================="

# Check if environment mode is specified
ENV_MODE=${1:-dev}
echo "📍 Environment: $ENV_MODE"

# Check if environment file exists
ENV_FILE=".env.$ENV_MODE"
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file not found: $ENV_FILE"
    echo "Available environment files:"
    ls -la .env.*
    exit 1
fi

echo "✅ Found environment file: $ENV_FILE"

# Load environment variables and show OAuth status
echo "🔍 Checking OneLogin OAuth configuration..."
source "$ENV_FILE"

# Check required OneLogin variables
REQUIRED_VARS=("OAUTH_ONELOGIN_CLIENT_ID" "OAUTH_ONELOGIN_CLIENT_SECRET" "OAUTH_ONELOGIN_DOMAIN" "OAUTH_ONELOGIN_REDIRECT_URI")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    else
        echo "✅ $var: configured"
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo "⚠️ WARNING: Missing OneLogin OAuth variables: ${MISSING_VARS[*]}"
    echo "⚠️ Application will run without OAuth authentication"
else
    echo "✅ All OneLogin OAuth variables are configured"
    echo "🌐 OAuth Domain: $OAUTH_ONELOGIN_DOMAIN"
    echo "🔗 Redirect URI: $OAUTH_ONELOGIN_REDIRECT_URI"
fi

echo ""
echo "🐳 Starting with Docker Compose..."

# Start services with proper environment file
docker compose --env-file "$ENV_FILE" -p compliance-chatbot up --build -d

# Show status
echo ""
echo "📊 Service Status:"
docker compose -p compliance-chatbot ps

echo ""
echo "🎯 Application URLs:"
if [ "$ENV_MODE" = "dev" ]; then
    echo "   Chatbot: http://localhost:8010/compliance/chat"
    echo "   Phoenix Tracing: http://localhost:3100"
    echo "   Redis Admin: http://localhost:8002"
else
    echo "   Chatbot: https://cpxis.global.lotuss.org/compliance/chat"
    echo "   Phoenix Tracing: http://localhost:3100"
    echo "   Redis Admin: http://localhost:8002"
fi

echo ""
echo "📋 Useful commands:"
echo "   View logs: docker compose -p compliance-chatbot logs -f"
echo "   Stop services: docker compose -p compliance-chatbot down"
echo "   Restart app: docker compose -p compliance-chatbot restart chainlit_app"

echo ""
echo "✅ Startup complete! Check the logs above for any errors."