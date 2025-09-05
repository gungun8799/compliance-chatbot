# AI-Powered Customer Chatbot

An intelligent chatbot system built with Chainlit, LlamaIndex, and various AI technologies for enhanced customer interaction and support.

## 🌟 Features

- **Multi-LLM Support**: Supports multiple language models including custom models and Groq
- **Vector Database Integration**: Uses Qdrant for efficient knowledge retrieval and semantic search
- **OAuth Authentication**: Secure access with OneLogin integration
- **Chat Memory**: Persistent chat history using Redis
- **Telemetry**: Performance monitoring with Phoenix
- **Hybrid Search**: Combines vector and sparse search for improved accuracy
- **Streaming Responses**: Real-time streaming of AI responses
- **Custom System Prompts**: Configurable conversation context and personality

## 🏗️ Architecture

The chatbot is built with the following key components:

- **Frontend**: Chainlit for the chat interface
- **Language Models**: 
  - Custom AI model via OpenAI-compatible API
  - Groq integration for alternative processing
- **Embeddings**: Text Embeddings Inference for semantic understanding
- **Vector Store**: Qdrant for efficient knowledge retrieval
- **Memory Store**: Redis for chat history persistence
- **Monitoring**: Phoenix for performance tracking
- **Authentication**: OneLogin OAuth provider

## 🔧 Technical Stack

- **Framework**: Chainlit
- **Vector Database**: Qdrant
- **Cache & Memory**: Redis
- **Embeddings**: Text Embeddings Inference
- **Authentication**: OneLogin OAuth
- **Monitoring**: Phoenix/OpenTelemetry
- **Database**: PostgreSQL with Prisma

## Prerequisites

- Docker installed on your machine.
- Docker Compose installed on your machine.
- A `.env` file with the necessary environment variables.
- Python and Chainlit installed (for local Chainlit app deployment).

---

## Deployment Steps

### 1. Create the Docker Network

First, create a Docker network named `ai`:

```bash
docker network create --driver bridge ai
```

### 2. Build and Deploy the Services

To build and deploy all services, run the following command:

```bash
docker compose -p customer_chatbot up --build
```

### 3. Build and Deploy Specific Services

If you need to build and deploy specific services, you can do so by specifying the service names. For example, to build and deploy the `chainlit_app`, `postgres`, and `prisma` services:

```bash
docker compose -p customer_chatbot up --build -d chainlit_app postgres prisma
```

### 4. Deploy SQL Schema and Start Prisma Studio

To deploy the SQL schema and start Prisma Studio, run:

```bash
docker compose -p customer_chatbot up --build -d postgres prisma
```

---

## Running Chainlit App Locally

If you want to run the Chainlit app locally without Docker, follow these steps:

1. **Copy `.env` to the `chainlit_app` folder**:
   - Copy the `.env` file to the `chainlit_app` directory.
   - Remove any lines in the `.env` file that start with `#` (comments) to ensure only valid environment variables are used.

2. **Navigate to the `chainlit_app` folder**:
   ```bash
   cd chainlit_app
   ```

3. **Run the Chainlit app**:
   Use the following command to start the Chainlit app:
   ```bash
   chainlit run app.py -h --root-path /chatbot/v1
   ```
   - The `-h` flag allows the app to be accessible on the network.
   - The `--root-path /chatbot/v1` sets the root path for the app.

4. **Access the Chainlit app**:
   Once the app is running, you can access it at `http://localhost:8000`.

---

## 📦 Screenshots

Below are some screenshots of the application:

### Main Page
![Main Page](screenshots/maing_page.png)


### Human Feedback
![Human Feedback](screenshots/humand_feedback.png)

---

## Accessing the Services

- **Phoenix**: Accessible at `http://localhost:3000`
- **Redis**: Accessible at `http://localhost:6379` (Redis) and `http://localhost:8001` (Redis Stack)
- **Qdrant**: Accessible at `http://localhost:6333` (REST API) and `http://localhost:6334` (gRPC API)
- **PostgreSQL**: Accessible at `http://localhost:5432`
- **Prisma Studio**: Accessible at `http://localhost:5555`
- **LocalStack**: Accessible at `http://localhost:4566`
- **Chainlit App**: Accessible at `http://localhost:8000`

---

## Environment Variables

Ensure you have a `.env` file with the necessary environment variables. Here are some of the key variables you might need:

```env
# Phoenix
PHOENIX_SECRET=your_secret_key
PHOENIX_ENABLE_AUTH=True
PHOENIX_USE_SECURE_COOKIES=True

# Redis
REDIS_CHATSTORE_PASSWORD=your_redis_password

# Qdrant
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_READ_ONLY_API_KEY=your_qdrant_read_only_api_key

# PostgreSQL
POSTGRES_DB=your_database_name
POSTGRES_USER=your_database_user
POSTGRES_PASSWORD=your_database_password

# Prisma
DATABASE_URL=postgresql://your_database_user:your_database_password@postgres:5432/your_database_name
```

---

## Volumes

The Docker Compose file uses local volumes for data persistence. You can modify the volume paths if needed, especially for production environments.

---

## Logging

Logs are configured to use the `json-file` driver with a maximum size of 10MB and up to 3 files per service.

---

## Networks

All services are connected to the `ai` network, which is created as an external network.

---

## Troubleshooting

- **Service Not Starting**: Ensure all required environment variables are set in the `.env` file.
- **Port Conflicts**: Check if the ports specified in the `docker-compose.yml` file are not already in use.
- **Volume Issues**: Ensure the volume paths are correct and accessible.

---

## Conclusion

This setup provides a robust environment for deploying a customer chatbot with various supporting services. Follow the steps above to get your chatbot up and running. For further customization, refer to the individual service documentation.

Happy deploying! 🚀