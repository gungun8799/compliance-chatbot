#!/bin/sh
set -e

# Default chainlit arguments
CHAINLIT_ARGS="run app.py --port 8010 --host 0.0.0.0"

# If in production, add the --root-path argument
if [ "$ENV_MODE" = "prod" ]; then
  CHAINLIT_ARGS="$CHAINLIT_ARGS --root-path /compliance-chat"
fi

# Execute chainlit with the constructed arguments
exec python -m chainlit $CHAINLIT_ARGS
