#!/usr/bin/env python3
"""
Script to fix vector dimension mismatch in Qdrant collections.
Run this when switching from 384-dim to 1024-dim embeddings.
"""
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Try loading different .env files
load_dotenv()  # Load from current dir
load_dotenv("../.env.prod")  # Load from parent dir
load_dotenv("../.env")  # Load from parent dir

def fix_collection_dimensions():
    """Recreate Qdrant collection with correct dimensions for bge-m3 (1024)"""
    # Use localhost URL for external access to Docker container  
    # Port mapping: 6433:6333 (HTTP) and 6434:6334 (gRPC)
    qdrant_url = "http://localhost:6433"  # HTTP port from your Docker mapping
    api_key = os.getenv("QDRANT_API_KEY")
    
    print(f"API Key loaded: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"API Key starts with: {api_key[:10]}...")
    
    client = QdrantClient(
        url=qdrant_url,
        api_key=api_key,
        https=False,
        prefer_grpc=False,  # Use HTTP instead of gRPC for external access
        check_compatibility=False,  # Skip version check
    )
    
    print(f"Connecting to Qdrant at: {qdrant_url}")
    
    # First, list all collections to see what exists
    try:
        collections = client.get_collections()
        print(f"Available collections: {[c.name for c in collections.collections]}")
        
        if not collections.collections:
            print("No collections found. The error might be due to missing collections.")
            return
            
        # Use the first collection or ask user to specify
        if len(collections.collections) == 1:
            collection_name = collections.collections[0].name
        else:
            print("Multiple collections found:")
            for i, col in enumerate(collections.collections):
                print(f"  {i+1}. {col.name}")
            choice = input("Enter collection number to check (or collection name): ").strip()
            
            if choice.isdigit():
                collection_name = collections.collections[int(choice)-1].name
            else:
                collection_name = choice
        
        print(f"Checking collection: {collection_name}")
        
        # Check existing collection
        collection_info = client.get_collection(collection_name=collection_name)
        vectors_config = collection_info.config.params.vectors
        
        # Handle different vector config formats (named vectors vs single vector)
        if isinstance(vectors_config, dict):
            # Named vectors configuration
            print(f"Named vectors found: {list(vectors_config.keys())}")
            
            # Check text-dense vector (commonly used for embeddings)
            if 'text-dense' in vectors_config:
                current_size = vectors_config['text-dense'].size
                print(f"'text-dense' vector size: {current_size}")
            else:
                # Get first vector config
                first_key = next(iter(vectors_config))
                current_size = vectors_config[first_key].size
                print(f"'{first_key}' vector size: {current_size}")
        else:
            current_size = vectors_config.size
            
        print(f"Vector config details: {vectors_config}")
        
        if current_size == 384:
            print("⚠️ Collection has 384 dimensions but bge-m3 produces 1024 dimensions")
            # For automated execution, default to yes
            response = "y"  # Change this to input() if you want manual confirmation
            print(f"Automatically proceeding to recreate collection: {response}")
            
            if response.lower() == 'y':
                # Delete existing collection
                print("🗑️ Deleting existing collection...")
                client.delete_collection(collection_name=collection_name)
                
                # Create new collection with correct dimensions (named vectors)
                print("🔧 Creating new collection with 1024 dimensions...")
                from qdrant_client.models import VectorParams, Distance
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "text-dense": VectorParams(size=1024, distance=Distance.COSINE)
                    },
                )
                print("✅ Collection recreated successfully!")
                print("⚠️ You'll need to re-index your documents with the new embeddings.")
            else:
                print("❌ Collection not modified. Change EMBED_MODEL_ID to a 384-dim model instead.")
        elif current_size == 1024:
            print("✅ Collection already has correct dimensions (1024)")
        else:
            print(f"⚠️ Unexpected vector size: {current_size}")
            
    except Exception as e:
        if "doesn't exist" in str(e).lower():
            print(f"Collection '{collection_name}' doesn't exist. Will be created automatically.")
        else:
            print(f"Error: {e}")

if __name__ == "__main__":
    fix_collection_dimensions()