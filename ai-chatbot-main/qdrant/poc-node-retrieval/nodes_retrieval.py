from llama_index.core.schema import QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
import httpx
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.vector_stores import VectorStoreQueryResult
import warnings
warnings.filterwarnings("ignore")
from patches import patch
# Apply the monkey patch
patch.apply_patch()

def relative_score_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,  # passed in from the query engine
    top_k: int = 2,  # passed in from the query engine i.e. similarity_top_k
) -> VectorStoreQueryResult:
    """
    Fuse dense and sparse results using relative score fusion.
    """
    # sanity check
    assert dense_result.nodes is not None
    assert dense_result.similarities is not None
    assert sparse_result.nodes is not None
    assert sparse_result.similarities is not None
    
    print("Using Hybrid Search:")
    print("top_k:", top_k)
    print("alpha:", alpha)
    print("Dense Nodes:", len(dense_result.nodes))
    print("Spare Nodes:", len(sparse_result.nodes))

    # deconstruct results
    sparse_result_tuples = list(
        zip(sparse_result.similarities, sparse_result.nodes)
    )
    sparse_result_tuples.sort(key=lambda x: x[0], reverse=True)

    dense_result_tuples = list(
        zip(dense_result.similarities, dense_result.nodes)
    )
    dense_result_tuples.sort(key=lambda x: x[0], reverse=True)

    # track nodes in both results
    all_nodes_dict = {x.node_id: x for x in dense_result.nodes}
    for node in sparse_result.nodes:
        if node.node_id not in all_nodes_dict:
            all_nodes_dict[node.node_id] = node

    # normalize sparse similarities from 0 to 1
    sparse_similarities = [x[0] for x in sparse_result_tuples]
    max_sparse_sim = max(sparse_similarities)
    min_sparse_sim = min(sparse_similarities)
    sparse_similarities = [
        (x - min_sparse_sim) / (max_sparse_sim - min_sparse_sim)
        for x in sparse_similarities
    ]
    sparse_per_node = {
        sparse_result_tuples[i][1].node_id: x
        for i, x in enumerate(sparse_similarities)
    }

    # normalize dense similarities from 0 to 1
    dense_similarities = [x[0] for x in dense_result_tuples]
    max_dense_sim = max(dense_similarities)
    min_dense_sim = min(dense_similarities)
    dense_similarities = [
        (x - min_dense_sim) / (max_dense_sim - min_dense_sim)
        for x in dense_similarities
    ]
    dense_per_node = {
        dense_result_tuples[i][1].node_id: x
        for i, x in enumerate(dense_similarities)
    }

    # fuse the scores
    fused_similarities = []
    for node_id in all_nodes_dict:
        sparse_sim = sparse_per_node.get(node_id, 0)
        dense_sim = dense_per_node.get(node_id, 0)
        fused_sim = alpha * (sparse_sim + dense_sim)
        fused_similarities.append((fused_sim, all_nodes_dict[node_id]))

    fused_similarities.sort(key=lambda x: x[0], reverse=True)
    fused_similarities = fused_similarities[:top_k]

    # create final response object
    return VectorStoreQueryResult(
        nodes=[x[1] for x in fused_similarities],
        similarities=[x[0] for x in fused_similarities],
        ids=[x[1].node_id for x in fused_similarities],
    )
    
# Create an httpx client with SSL verification disabled
http_client = httpx.Client(verify=False)
# Create an httpx AsyncClient with SSL verification using certifi
async_http_client = httpx.AsyncClient(verify=False)

# Set up LLM model
Settings.llm = OpenAILike(
    model="default",
    api_base="https://api.cpxis.global.lotuss.org/llm/v1",
    api_key="automation.lotuss.Zb71t4pjNR3rty3uI8os9jwxaJmU8h",
    is_chat_model=True,
    is_function_calling_model=False,
    temperature=0.2,
    http_client=http_client,
    async_http_client=async_http_client,
    # max_tokens=4096,
)

# Initialize the embedding settings
Settings.embed_model = TextEmbeddingsInference(
    model_name="BAAI/bge-m3",
    base_url=f"https://api.cpxis.global.lotuss.org/embedding/BAAI/bge-m3",
    auth_token=f"Bearer automation.lotuss.Zb71t4pjNR3rty3uI8os9jwxaJmU8h",
    timeout=60,
    embed_batch_size=10,
)

api_key = "QdrantVAsfhF8nGPtyleJKVkt2TBI2bqQ4bSjgnajNtOLwE2Y9YWxnZFrItRBE53"
# creates a persistant index to disk
client = QdrantClient(url="http://localhost:6334", api_key=api_key,  prefer_grpc=True)
aclient = AsyncQdrantClient(url="http://localhost:6334", api_key=api_key, prefer_grpc=True)

# create our vector store with hybrid indexing enabled
# batch_size controls how many nodes are encoded with sparse vectors at once
vector_store = QdrantVectorStore(
    "vector_data",
    client=client,
    aclient=aclient,
    enable_hybrid=True,
    batch_size=20,
    prefer_grpc=True,
    hybrid_fusion_fn=relative_score_fusion,
)

# Assume you have a VectorStoreIndex instance
index = VectorStoreIndex.from_vector_store(vector_store)

# Initialize the VectorIndexRetriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
    sparse_top_k=12,
    alpha=0.5,
    vector_store_query_mode="hybrid" 
)

# Create a QueryBundle with your query and context
query_bundle = QueryBundle(
    query_str="What is Lotus's coin?",
    # context_str="Focus on environmental and economic impacts."
)

# Use the retriever to get the most relevant document nodes
nodes_with_scores = retriever._retrieve(query_bundle)

# Print the retrieved nodes and their scores
for node_with_score in nodes_with_scores:
    print()
    print(f"Node: {node_with_score.node}")
    print(f"Score: {node_with_score.score}")
