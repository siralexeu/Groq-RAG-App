import re
import chromadb
from chromadb.config import Settings
import streamlit as st


# Folosește ChromaDB in-memory pentru Streamlit Cloud
# Datele persistă pe durata sesiunii aplicației
@st.cache_resource
def get_chroma_client():
    """
    Create an in-memory ChromaDB client with explicit settings
    to prevent SQLite persistence issues.
    """
    return chromadb.EphemeralClient(
        settings=Settings(
            anonymized_telemetry=False,
            is_persistent=False
        )
    )


chroma_client = get_chroma_client()


def sanitize_collection_name(name):
    """
    Sanitize collection name to meet ChromaDB requirements:
    - Only alphanumeric, underscore, and dash
    - Maximum 63 characters
    - Must start and end with alphanumeric
    """
    # Replace invalid characters with underscore
    sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    # Ensure it starts with alphanumeric
    sanitized_name = re.sub(r"^[^a-zA-Z0-9]+", "", sanitized_name)

    # Ensure it ends with alphanumeric
    sanitized_name = re.sub(r"[^a-zA-Z0-9]+$", "", sanitized_name)

    # Ensure minimum length of 3 characters
    if len(sanitized_name) < 3:
        sanitized_name = f"col_{sanitized_name}"

    # Truncate to 63 characters
    sanitized_name = sanitized_name[:63]

    # Make sure it still ends with alphanumeric after truncation
    sanitized_name = re.sub(r"[^a-zA-Z0-9]+$", "", sanitized_name)

    return sanitized_name.lower()


def create_or_get_collection(collection_name):
    """
    Create or retrieve a collection from ChromaDB.
    Handles existing collections gracefully.
    """
    try:
        # Try to get existing collection
        collection = chroma_client.get_collection(collection_name)
        return collection
    except Exception as e:
        # Collection doesn't exist, create it
        try:
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return collection
        except Exception as create_error:
            # If creation fails, try to get it again (race condition)
            try:
                return chroma_client.get_collection(collection_name)
            except:
                st.error(f"Failed to create or get collection '{collection_name}': {create_error}")
                raise


def add_documents_to_collection(collection, chunks, embeddings):
    """
    Add documents and embeddings to a collection.
    Handles duplicates gracefully.
    """
    added_count = 0
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        try:
            collection.add(
                ids=[f"chunk_{idx}"],
                documents=[chunk],
                metadatas=[{"chunk_index": idx}],
                embeddings=[embedding],
            )
            added_count += 1
        except Exception as e:
            # Chunk already exists or other error - skip silently
            continue

    return added_count


def query_collection(collection, query_embedding, n_results=3):
    """
    Query the collection with an embedding and return relevant documents.
    """
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        if "documents" in results and results["documents"]:
            return results["documents"][0]
    except Exception as e:
        st.warning(f"Error querying collection: {e}")
    return []


def delete_collection(collection_name):
    """
    Delete a collection from ChromaDB.
    """
    try:
        chroma_client.delete_collection(name=collection_name)
        return True
    except Exception as e:
        st.warning(f"Error deleting collection '{collection_name}': {e}")
        return False


def list_all_collections():
    """
    List all collections in ChromaDB.
    """
    try:
        collections = chroma_client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        st.warning(f"Error listing collections: {e}")
        return []


def collection_exists(collection_name):
    """
    Check if a collection exists.
    """
    try:
        chroma_client.get_collection(collection_name)
        return True
    except:
        return False


def get_collection_count(collection):
    """
    Get the number of documents in a collection.
    """
    try:
        return collection.count()
    except Exception as e:
        st.warning(f"Error getting collection count: {e}")
        return 0