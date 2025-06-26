import redis
import json
import os
from datetime import datetime

# === Redis Config ===
REDIS_HOST = "redis"
REDIS_PORT = 6379
REDIS_PASSWORD = "u@U5410154"

# === Output Directory ===
OUTPUT_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Connect to Redis ===
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)

# === Fetch all conversation logs ===
keys = r.keys("conversation_log:*")

print(f"🔍 Found {len(keys)} conversation logs...")

for key in keys:
    thread_id = key.split(":")[1]
    try:
        raw = r.get(key)
        if not raw:
            continue

        messages = json.loads(raw)

        # Format readable text
        lines = []
        for msg in messages:
            timestamp = datetime.fromtimestamp(msg["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
            role = msg["role"]
            content = msg["content"]
            difficulty = msg.get("difficulty", "")

            if role == "bot":
                lines.append(f"[{timestamp}] 🤖 BOT ({difficulty}): {content}")
            elif role == "user":
                lines.append(f"[{timestamp}] 👤 USER: {content}")
            else:
                lines.append(f"[{timestamp}] 🛠️ {role.upper()}: {content}")

        # Save to .txt file
        output_path = os.path.join(OUTPUT_DIR, f"{thread_id}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(lines))

        print(f"✅ Exported: {output_path}")

    except Exception as e:
        print(f"❌ Error reading {key}: {e}")