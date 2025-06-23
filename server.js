import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import { createClient } from 'redis';

const app = express();
const PORT = 4001;

// ✅ Connect to Redis
const redisClient = createClient({
  url: 'redis://:u%40U5410154@localhost:6379'
});
await redisClient.connect();

app.use(cors());
app.use(bodyParser.json());


app.post('/api/feedback', async (req, res) => {
  try {
    const { actionName, parentId, sessionId } = req.body;

    if (!actionName?.startsWith('feedback_')) {
      return res.status(400).json({ message: 'Invalid feedback action' });
    }

    const score = parseInt(actionName.split('_', 2)[1], 10);
    if (isNaN(score) || score < 1 || score > 5) {
      return res.status(400).json({ message: 'Invalid score value' });
    }

    // Push into Redis list
    const entry = JSON.stringify({
      timestamp: Date.now(),
      score,
    });
    await redisClient.rPush(`feedback:${sessionId}:${parentId}`, entry);

    return res.json({ message: `🙏 Thanks for your feedback: **${score}/5**` });
  } catch (e) {
    console.error('Error saving feedback:', e);
    return res.status(500).json({ message: 'Server error' });
  }
});



// ✅ Receive admin reply from Power Automate
app.post('/api/admin-reply', async (req, res) => {
  const { thread_id, parent_id, parent_content, replies } = req.body;

  console.log('📨 Incoming POST /api/admin-reply:', JSON.stringify(req.body, null, 2));

  if (!thread_id || !parent_id || !parent_content || !Array.isArray(replies)) {
    return res.status(400).json({ error: 'Missing or invalid thread_id, parent_id, parent_content, or replies' });
  }

  const key = `admin-reply:${thread_id}:${parent_id}`;

  try {
    // Retrieve existing replies from Redis
    let existingPayload = {};
    const existing = await redisClient.get(key);
    if (existing) {
      existingPayload = JSON.parse(existing);
    }

    const existingReplyIds = new Set(
      (existingPayload.replies || []).map(r => r.id)
    );

    // Deduplicate and merge new replies
    const newReplies = replies
      .filter(r => r?.id && !existingReplyIds.has(r.id))
      .sort((a, b) => new Date(a.createdDateTime) - new Date(b.createdDateTime));

    if (newReplies.length === 0) {
      console.log(`⚠️ No new replies to store for ${key}`);
      return res.json({ status: 'no-update' });
    }

    const mergedReplies = [...(existingPayload.replies || []), ...newReplies];

    const valueToStore = JSON.stringify({
      parent_content,
      replies: mergedReplies
    });

    await redisClient.setEx(key, 3600, valueToStore); // TTL: 1 hour
    console.log(`✅ Stored Redis key: ${key} with ${mergedReplies.length} replies`);

    return res.json({ status: 'ok' });

  } catch (err) {
    console.error('❌ Redis error:', err);
    return res.status(500).json({ error: 'Redis failed' });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Admin reply server running at http://localhost:${PORT}`);
});