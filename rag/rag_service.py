"""
RAG Service
Main logic for document retrieval and response generation
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from rag.document_parser import TikaDocumentParser
from rag.text_chunker import TextChunker, Document
from rag.embeddings import BGEEmbeddings
from rag.vector_store import MilvusVectorStore


class RAGService:
    """Main RAG service"""
    
    def __init__(self,
                 groq_api_key: Optional[str] = None,
                 milvus_uri: str = "http://localhost:19530",
                 tika_url: str = "http://localhost:9998",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_device: str = None,
                 groq_model_name: str = "llama-3.1-70b-versatile",
                 word_limit: int = 150):
        """
        Initialize RAG service
        
        Args:
            groq_api_key: Groq API key (optional - only needed for LLM responses)
            milvus_uri: Milvus URI
            tika_url: Tika server URL
            embedding_model: HuggingFace embedding model name
            embedding_device: Device for embeddings (cuda/cpu/mps, auto-detect if None)
            groq_model_name: Groq model to use
            word_limit: Word limit for chunking
        """
        self.parser = TikaDocumentParser(tika_url=tika_url)
        self.chunker = TextChunker(word_limit=word_limit)
        self.embeddings = BGEEmbeddings(
            model_name=embedding_model,
            device=embedding_device
        )
        self.vector_store = MilvusVectorStore(uri=milvus_uri)
        
        # LLM is optional - only for generating responses (not needed for context retrieval)
        self.llm = None
        if groq_api_key:
            from core.groq_llm import GroqLLM
            self.llm = GroqLLM(api_key=groq_api_key, model_name=groq_model_name)
        
        # Connect to Milvus
        self.vector_store.connect()
    
    def process_document(self,
                        file_path: str,
                        file_id: str,
                        file_name: str,
                        collection_name: str = "documents") -> Dict[str, Any]:
        """
        Process and index a document
        
        Args:
            file_path: Path to document
            file_id: Unique file ID
            file_name: Original filename
            collection_name: Collection to store in
            
        Returns:
            Processing statistics
        """
        try:
            logger.info(f"Processing document: {file_name}")
            
            # Parse document
            text = self.parser.parse_document(file_path)
            if not text:
                raise ValueError("Failed to extract text from document")
            
            # Chunk text
            chunks = self.chunker.chunk_text(
                text=text,
                metadata={
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path
                }
            )
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Create collection if needed
            # Create collection if needed
            self.vector_store.create_collection(
                collection_name=collection_name, 
                dimension=self.embeddings.dimension
            )
            
            # Insert documents
            stats = self.vector_store.insert_documents(
                collection_name=collection_name,
                documents=chunks,
                embeddings_model=self.embeddings
            )
            
            logger.info(f"Document processed successfully: {file_name}")
            
            return {
                "file_id": file_id,
                "file_name": file_name,
                "chunks_created": len(chunks),
                "embedding_tokens": stats["embedding_tokens"],
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def query(self,
             query_text: str,
             collection_name: str = "documents",
             top_k: int = 5,
             file_ids: Optional[List[str]] = None,
             conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Query documents and generate response
        
        Args:
            query_text: User query
            collection_name: Collection to search
            top_k: Number of documents to retrieve
            file_ids: Filter by file IDs
            conversation_history: Previous conversation
            
        Returns:
            Response with sources and tokens
        """
        try:
            logger.info(f"Processing query: {query_text[:50]}...")
            
            # Search for relevant documents
            results = self.vector_store.search(
                collection_name=collection_name,
                query_text=query_text,
                embeddings_model=self.embeddings,
                top_k=top_k,
                file_ids=file_ids
            )
            
            # Build context from results
            context_parts = []
            sources = []
            
            for idx, result in enumerate(results, 1):
                context_parts.append(f"[Source {idx}] {result['text']}")
                sources.append({
                    "file_id": result["file_id"],
                    "file_name": result["file_name"],
                    "chunk_number": result["chunk_number"],
                    "page_number": result["page_number"],
                    "score": result["score"]
                })
            
            context = "\n\n".join(context_parts)
            
            # Build prompt
            system_prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

Context from documents:
{context}

Instructions:
- Answer based on the context provided
- If the context doesn't contain relevant information, say so
- Be concise and accurate
- Cite sources when possible (e.g., "According to Source 1...")
"""
            
            # Set system instruction
            self.llm.set_system_instruction(system_prompt)
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history:
                    if msg.get("role") == "user":
                        self.llm.add_to_history("user", msg.get("content", ""))
                    elif msg.get("role") == "assistant":
                        self.llm.add_to_history("assistant", msg.get("content", ""))
            
            # Generate response
            response = self.llm.generate_response(query_text)
            
            logger.info("Generated response successfully")
            
            return {
                "response": response,
                "sources": sources,
                "tokens": {
                    "embedding_tokens": self.embeddings.query_tokens,
                    "llm_tokens": "tracked_by_groq"
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    def retrieve_context(self, query_text: str, top_k: int = 5, collection_name: str = "documents") -> str:
        """
        Retrieve context for a query without generating a response
        """
        try:
            results = self.vector_store.search(
                collection_name=collection_name,
                query_text=query_text,
                embeddings_model=self.embeddings,
                top_k=top_k
            )
            
            context_parts = []
            for idx, result in enumerate(results, 1):
                context_parts.append(f"[Source {idx}] {result['text']}")
            
            return "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return f"Error retrieving context: {str(e)}"

    async def query_stream(self,
                           query_text: str,
                           collection_name: str = "documents",
                           top_k: int = 5):
        """
        Stream RAG response for low latency voice applications.
        
        Yields text chunks as they are generated.
        """
        try:
            logger.info(f"[RAG STREAM] Processing: {query_text[:50]}...")
            
            # Retrieve context (synchronous - fast)
            context = self.retrieve_context(query_text, top_k, collection_name)
            
            # Build RAG prompt
            system_prompt = f"""You are a helpful voice assistant. Answer based on the provided context.

Context from documents:
{context}

Instructions:
- Answer based on the context provided
- Be concise and natural for speech
- If the context doesn't help, answer generally
"""
            
            # Set system prompt and stream response
            self.llm.set_system_instruction(system_prompt)
            
            async for chunk in self.llm.generate_response_stream(query_text):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in query_stream: {e}")
            yield f"Sorry, I encountered an error: {str(e)}"
