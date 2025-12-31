"""
HuggingFace BGE Embeddings Wrapper
Handles text embeddings using BAAI/bge-large-en-v1.5
"""
from typing import List
from sentence_transformers import SentenceTransformer
from loguru import logger
import torch


class BGEEmbeddings:
    """HuggingFace BGE embeddings with token tracking"""
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-large-en-v1.5",
                 device: str = None,
                 batch_size: int = 32):
        """
        Initialize BGE embeddings
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda, cpu, mps). Auto-detect if None
            batch_size: Batch size for embedding
        """
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        # Load model
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Embedding model loaded successfully")
        
        # Get dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.is_bge = "bge" in model_name.lower()
        
        # Token tracking (approximate based on text length)
        self.query_tokens = 0
        self.document_tokens = 0
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)"""
        return len(text) // 4
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Add instruction for document embedding (Only for BGE models)
            if self.is_bge:
                final_texts = [f"Represent this document for retrieval: {text}" for text in texts]
            else:
                final_texts = texts
            
            # Generate embeddings
            embeddings = self.model.encode(
                final_texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=False,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            # Convert to list of lists
            embeddings_list = [emb.tolist() for emb in embeddings]
            
            # Track tokens (approximate)
            tokens = sum(self._estimate_tokens(text) for text in texts)
            self.document_tokens += tokens
            
            logger.info(f"Embedded {len(texts)} documents, estimated tokens: {tokens}")
            
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed query text
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        try:
            # Add instruction for query embedding (Only for BGE models)
            if self.is_bge:
                final_text = f"Represent this query for retrieving relevant documents: {text}"
            else:
                final_text = text
            
            # Generate embedding
            embedding = self.model.encode(
                final_text,
                convert_to_numpy=False,
                normalize_embeddings=True
            )
            
            # Track tokens (approximate)
            tokens = self._estimate_tokens(text)
            self.query_tokens += tokens
            
            logger.debug(f"Embedded query, estimated tokens: {tokens}")
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def get_total_tokens(self) -> int:
        """Get total tokens used (approximate)"""
        return self.query_tokens + self.document_tokens
    
    def reset_tokens(self):
        """Reset token counters"""
        self.query_tokens = 0
        self.document_tokens = 0
