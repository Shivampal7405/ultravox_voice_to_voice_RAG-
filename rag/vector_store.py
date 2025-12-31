"""
Milvus Vector Store Operations
Handles vector database connections and operations
"""
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from loguru import logger
from rag.text_chunker import Document
from rag.embeddings import BGEEmbeddings


class MilvusVectorStore:
    """Milvus vector database operations"""
    
    def __init__(self,
                 uri: str = "http://localhost:19530",
                 db_name: str = "voice_rag",
                 token: str = ""):
        """
        Initialize Milvus connection
        
        Args:
            uri: Milvus server URI
            db_name: Database name
            token: Authentication token (if needed)
        """
        self.uri = uri
        self.db_name = db_name
        self.token = token
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to Milvus"""
        try:
            connections.connect(
                alias="default",
                uri=self.uri,
                token=self.token
            )
            self.connected = True
            logger.info(f"Connected to Milvus at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False
    
    def drop_collection(self, collection_name: str) -> bool:
        """Drop a collection by name"""
        try:
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info(f"Dropped collection: {collection_name}")
                return True
            else:
                logger.info(f"Collection {collection_name} does not exist")
                return False
        except Exception as e:
            logger.error(f"Failed to drop collection: {e}")
            return False
    
    def create_collection(self,
                         collection_name: str,
                         dimension: int = 1024,
                         drop_existing: bool = False) -> bool:
        """
        Create a collection
        
        Args:
            collection_name: Name of collection
            dimension: Embedding dimension
            drop_existing: Drop if exists
            
        Returns:
            True if successful
        """
        try:
            if utility.has_collection(collection_name):
                # Check if dimension matches
                existing_coll = Collection(collection_name)
                existing_dim = -1
                for field in existing_coll.schema.fields:
                    if field.name == "embedding":
                        existing_dim = field.params.get("dim")
                        break
                
                if existing_dim != dimension:
                    logger.warning(f"Collection {collection_name} dimension mismatch (Existing: {existing_dim}, New: {dimension}). Dropping and recreating.")
                    utility.drop_collection(collection_name)
                elif drop_existing:
                    utility.drop_collection(collection_name)
                    logger.info(f"Dropped existing collection: {collection_name}")
                else:
                    logger.info(f"Collection already exists: {collection_name}")
                    return True
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunk_number", dtype=DataType.INT64),
                FieldSchema(name="page_number", dtype=DataType.INT64),
            ]
            
            schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name}")
            collection = Collection(name=collection_name, schema=schema)
            
            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            
            logger.info(f"Created collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def insert_documents(self,
                        collection_name: str,
                        documents: List[Document],
                        embeddings_model: BGEEmbeddings) -> Dict[str, Any]:
        """
        Insert documents into collection
        
        Args:
            collection_name: Collection name
            documents: List of documents
            embeddings_model: Embeddings model
            
        Returns:
            Insert statistics
        """
        try:
            # Get embeddings
            texts = [doc.page_content for doc in documents]
            embeddings = embeddings_model.embed_documents(texts)
            
            # Prepare data
            data = []
            for doc, embedding in zip(documents, embeddings):
                data.append({
                    "embedding": embedding,
                    "text": doc.page_content,
                    "file_id": doc.metadata.get("file_id", ""),
                    "file_name": doc.metadata.get("file_name", ""),
                    "chunk_number": doc.metadata.get("chunk_number", 0),
                    "page_number": doc.metadata.get("page_number", 0),
                })
            
            # Insert
            collection = Collection(collection_name)
            collection.insert(data)
            collection.flush()
            
            logger.info(f"Inserted {len(documents)} documents into {collection_name}")
            
            return {
                "inserted_count": len(documents),
                "embedding_tokens": embeddings_model.document_tokens
            }
            
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise
    
    def search(self,
              collection_name: str,
              query_text: str,
              embeddings_model: BGEEmbeddings,
              top_k: int = 5,
              file_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            collection_name: Collection name
            query_text: Query text
            embeddings_model: Embeddings model
            top_k: Number of results
            file_ids: Filter by file IDs
            
        Returns:
            List of search results
        """
        try:
            # Get query embedding
            query_embedding = embeddings_model.embed_query(query_text)
            
            # Load collection
            collection = Collection(collection_name)
            collection.load()
            
            # Build filter expression
            expr = None
            if file_ids:
                expr = " or ".join([f'file_id == "{fid}"' for fid in file_ids])
            
            # Search
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["text", "file_id", "file_name", "chunk_number", "page_number"]
            )
            
            # Format results
            formatted_results = []
            for hit in results[0]:
                formatted_results.append({
                    "text": hit.entity.get("text"),
                    "file_id": hit.entity.get("file_id"),
                    "file_name": hit.entity.get("file_name"),
                    "chunk_number": hit.entity.get("chunk_number"),
                    "page_number": hit.entity.get("page_number"),
                    "score": hit.score
                })
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from Milvus"""
        try:
            connections.disconnect("default")
            self.connected = False
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
