-- 1. Create the users table (Chainlit expects the name: users)
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY,
  identifier TEXT UNIQUE NOT NULL,
  "createdAt" TEXT DEFAULT CURRENT_TIMESTAMP::TEXT,
  "updatedAt" TEXT DEFAULT CURRENT_TIMESTAMP::TEXT,
  metadata JSONB NOT NULL
);

-- 2. Insert default admin user
INSERT INTO users (id, identifier, "createdAt", "updatedAt", metadata)
SELECT
  gen_random_uuid(),
  'admin',
  CURRENT_TIMESTAMP::TEXT,
  CURRENT_TIMESTAMP::TEXT,
  '{"role": "ADMIN", "email": "chatbot_admin@gmail.com", "provider": "credentials"}'::jsonb
WHERE NOT EXISTS (
  SELECT 1 FROM users WHERE identifier = 'admin'
);

-- 3. Create threads table
CREATE TABLE IF NOT EXISTS threads (
  id UUID PRIMARY KEY,
  "createdAt" TEXT DEFAULT CURRENT_TIMESTAMP::TEXT,
  "updatedAt" TEXT DEFAULT CURRENT_TIMESTAMP::TEXT,
  "userId" UUID,
  "userIdentifier" TEXT,
  name TEXT,
  tags TEXT[] DEFAULT '{}',
  metadata JSONB NOT NULL DEFAULT '{}'
);

-- 4. Insert default thread (optional)
INSERT INTO threads (id, "createdAt", "updatedAt", metadata)
VALUES (
  gen_random_uuid(),
  CURRENT_TIMESTAMP::TEXT,
  CURRENT_TIMESTAMP::TEXT,
  '{"created_by": "system", "note": "default thread"}'::jsonb
)
ON CONFLICT DO NOTHING;

-- 5. Create clarification state
CREATE TABLE IF NOT EXISTS clarification_state (
  thread_id TEXT PRIMARY KEY,
  summaries JSON NOT NULL,
  nodes JSON NOT NULL
);

-- 6. Create steps table
CREATE TABLE IF NOT EXISTS steps (
  id UUID PRIMARY KEY,
  "threadId" UUID,
  "parentId" UUID,
  "createdAt" TEXT DEFAULT CURRENT_TIMESTAMP::TEXT,
  "updatedAt" TEXT DEFAULT CURRENT_TIMESTAMP::TEXT,
  "start" TEXT,
  "end" TEXT,
  input TEXT,
  output TEXT,
  name TEXT,
  type TEXT NOT NULL,
  streaming BOOLEAN,
  "isError" BOOLEAN DEFAULT FALSE,
  "waitForAnswer" BOOLEAN,
  "showInput" TEXT DEFAULT 'json',
  metadata JSONB NOT NULL DEFAULT '{}',
  generation JSONB NOT NULL DEFAULT '{}',
  tags TEXT[] DEFAULT '{}',
  language TEXT
);

CREATE TABLE IF NOT EXISTS elements (
  id UUID PRIMARY KEY,
  "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  "threadId" UUID,
  "stepId" UUID NOT NULL,
  "forId" UUID,                     
  "type" TEXT,                     
  metadata JSONB DEFAULT '{}',
  mime TEXT,
  name TEXT,
  "objectKey" TEXT,
  url TEXT,
  "chainlitKey" TEXT,
  display TEXT,
  size TEXT,                       
  language TEXT,
  page INTEGER,
  props JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS feedbacks (
  id UUID PRIMARY KEY,
  "threadId" UUID,
  "stepId" UUID,
  "forId" UUID,  -- Required for Chainlit join
  name TEXT,
  value FLOAT NOT NULL,
  comment TEXT,
  metadata JSONB NOT NULL DEFAULT '{}',
  "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add indexes similar to Prisma
CREATE INDEX IF NOT EXISTS idx_feedback_createdAt ON feedbacks ("createdAt");
CREATE INDEX IF NOT EXISTS idx_feedback_name ON feedbacks (name);
CREATE INDEX IF NOT EXISTS idx_feedback_stepId ON feedbacks ("stepId");
CREATE INDEX IF NOT EXISTS idx_feedback_value ON feedbacks (value);
CREATE INDEX IF NOT EXISTS idx_feedback_name_value ON feedbacks (name, value);