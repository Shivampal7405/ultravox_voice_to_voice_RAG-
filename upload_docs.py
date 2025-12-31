"""
Document Upload Script for RAG
Upload documents to Milvus vector store
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from rag.rag_service import RAGService
from core.config import get_rag_settings
import uuid


def upload_document(file_path: str, collection_name: str = None):
    """
    Upload a document to RAG
    
    Args:
        file_path: Path to document file
        collection_name: Optional collection name (default from config)
    """
    settings = get_rag_settings()
    collection = collection_name or settings.collection
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    file_name = os.path.basename(file_path)
    file_id = str(uuid.uuid4())[:8]
    
    print(f"\n📄 Uploading: {file_name}")
    print(f"   Collection: {collection}")
    print(f"   File ID: {file_id}")
    print()
    
    try:
        # Initialize RAG service
        print("🔄 Initializing RAG service...")
        rag = RAGService(
            groq_api_key=settings.groq_api_key,
            milvus_uri=settings.milvus_uri,
            embedding_model=settings.embedding_model,
            groq_model_name=settings.groq_model
        )
        
        # Process document
        print("🔄 Processing document...")
        result = rag.process_document(
            file_path=file_path,
            file_id=file_id,
            file_name=file_name,
            collection_name=collection
        )
        
        print(f"\n✅ Upload successful!")
        print(f"   Chunks created: {result['chunks_created']}")
        print(f"   Embedding tokens: {result['embedding_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        logger.exception("Upload error")
        return False


def upload_text(text: str, source_name: str = "manual_input", collection_name: str = None):
    """
    Upload raw text to RAG
    
    Args:
        text: Text content
        source_name: Name for this text source
        collection_name: Optional collection name
    """
    settings = get_rag_settings()
    collection = collection_name or settings.collection
    
    file_id = str(uuid.uuid4())[:8]
    
    print(f"\n📝 Uploading text: {source_name}")
    print(f"   Collection: {collection}")
    print(f"   Length: {len(text)} characters")
    print()
    
    try:
        # Initialize components
        from rag.text_chunker import TextChunker
        from rag.embeddings import BGEEmbeddings
        from rag.vector_store import MilvusVectorStore
        
        print("🔄 Initializing components...")
        chunker = TextChunker(word_limit=150)
        embeddings = BGEEmbeddings(model_name=settings.embedding_model)
        vector_store = MilvusVectorStore(uri=settings.milvus_uri)
        vector_store.connect()
        
        # Create collection
        vector_store.create_collection(collection, dimension=embeddings.dimension)
        
        # Chunk text
        print("🔄 Chunking text...")
        chunks = chunker.chunk_text(
            text=text,
            metadata={
                "file_id": file_id,
                "file_name": source_name,
                "file_path": "manual_input"
            }
        )
        
        # Insert
        print("🔄 Embedding and storing...")
        stats = vector_store.insert_documents(
            collection_name=collection,
            documents=chunks,
            embeddings_model=embeddings
        )
        
        print(f"\n✅ Upload successful!")
        print(f"   Chunks created: {len(chunks)}")
        print(f"   Embedding tokens: {stats['embedding_tokens']}")
        
        vector_store.disconnect()
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        logger.exception("Upload error")
        return False


def test_search(query: str, collection_name: str = None):
    """Test RAG search"""
    settings = get_rag_settings()
    collection = collection_name or settings.collection
    
    print(f"\n🔍 Testing search: '{query}'")
    print(f"   Collection: {collection}")
    print()
    
    try:
        rag = RAGService(
            groq_api_key=settings.groq_api_key,
            milvus_uri=settings.milvus_uri,
            embedding_model=settings.embedding_model,
            groq_model_name=settings.groq_model
        )
        
        context = rag.retrieve_context(
            query_text=query,
            top_k=settings.top_k,
            collection_name=collection
        )
        
        if context and not context.startswith("Error"):
            print("📚 Results:")
            print("-" * 50)
            print(context)
            print("-" * 50)
        else:
            print("❌ No results found")
            
    except Exception as e:
        print(f"❌ Search failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload documents to RAG system"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Upload file command
    upload_parser = subparsers.add_parser("upload", help="Upload a document file")
    upload_parser.add_argument("file", help="Path to document file")
    upload_parser.add_argument("--collection", "-c", help="Collection name")
    
    # Upload text command
    text_parser = subparsers.add_parser("text", help="Upload raw text")
    text_parser.add_argument("text", help="Text content (or @filename to read from file)")
    text_parser.add_argument("--name", "-n", default="manual_input", help="Source name")
    text_parser.add_argument("--collection", "-c", help="Collection name")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Test search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--collection", "-c", help="Collection name")
    
    args = parser.parse_args()
    
    if args.command == "upload":
        upload_document(args.file, args.collection)
        
    elif args.command == "text":
        text = args.text
        # Read from file if starts with @
        if text.startswith("@"):
            with open(text[1:], "r", encoding="utf-8") as f:
                text = f.read()
        upload_text(text, args.name, args.collection)
        
    elif args.command == "search":
        test_search(args.query, args.collection)
        
    else:
        parser.print_help()
        print("\n📌 Examples:")
        print("  python upload_docs.py upload document.pdf")
        print("  python upload_docs.py text 'Your knowledge text here'")
        print("  python upload_docs.py text @knowledge.txt --name 'Product Info'")
        print("  python upload_docs.py search 'What is the product?'")


if __name__ == "__main__":
    main()
