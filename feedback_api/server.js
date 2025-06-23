// feedback_api/server.js
import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import { createClient } from 'redis';

const app = express();        // ← must be here before any app.* calls
const PORT = 4001;

// ✅ Connect to Redis
const redisClient = createClient({
  url: 'redis://:u%40U5410154@redis:6379'  // use service name “redis” on the Docker network
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

    const entry = JSON.stringify({ timestamp: Date.now(), score });
    await redisClient.rPush(`feedback:${sessionId}:${parentId}`, entry);

    return res.json({ message: `🙏 Thanks for your feedback: **${score}/5**` });
  } catch (e) {
    console.error('Error saving feedback:', e);
    return res.status(500).json({ message: 'Server error' });
  }
});

app.post('/api/admin-reply', async (req, res) => {
  /* … your existing admin‐reply handler … */
});

app.listen(PORT, () => {
  console.log(`🚀 Feedback API listening on http://0.0.0.0:${PORT}`);
});