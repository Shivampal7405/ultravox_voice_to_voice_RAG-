"""
Text Chunker using spaCy
Splits text into chunks based on sentences and word limits
"""
from typing import List, Dict, Any
import re
import spacy
import subprocess
import sys
from loguru import logger


# Document class (simple dataclass)
class Document:
    """Simple document class for RAG"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}
    
    def dict(self):
        return {
            "page_content": self.page_content,
            "metadata": self.metadata
        }


def load_spacy_model(model_name="en_core_web_sm", max_length=9_999_999):
    """Load spaCy model, download if not available"""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        logger.info(f"spaCy model '{model_name}' not found. Downloading...")
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
        nlp = spacy.load(model_name)
    
    nlp.max_length = max_length
    return nlp


# Load spaCy model globally
try:
    nlp = load_spacy_model()
    logger.info("spaCy model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    nlp = None


class TextChunker:
    """Text chunking using spaCy sentence detection"""
    
    def __init__(self, word_limit: int = 150, min_chunk_length: int = 20):
        """
        Initialize text chunker
        
        Args:
            word_limit: Maximum words per chunk
            min_chunk_length: Minimum characters for a valid chunk
        """
        self.word_limit = word_limit
        self.min_chunk_length = min_chunk_length
        
        if nlp is None:
            raise RuntimeError("spaCy model not loaded. Cannot initialize TextChunker.")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before chunking
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove citation markers like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        return text.strip()
    
    def chunk_text_by_words(self, text: str) -> List[str]:
        """
        Chunk text based on sentences and word limit
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        doc = nlp(text)
        chunks = []
        current_chunk = []
        word_count = 0
        
        for sent in doc.sents:
            words = sent.text.split()
            word_count += len(words)
            current_chunk.append(sent.text)
            
            if word_count >= self.word_limit:
                chunk = " ".join(current_chunk)
                if len(chunk) >= self.min_chunk_length:
                    chunks.append(chunk)
                current_chunk = []
                word_count = 0
        
        # Add remaining chunk
        if current_chunk:
            chunk = " ".join(current_chunk)
            if len(chunk) >= self.min_chunk_length:
                chunks.append(chunk)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(chunks))
    
    def chunk_text(self, 
                   text: str, 
                   metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Split text into chunks and create Document objects
        
        Args:
            text: Text to chunk
            metadata: Base metadata to attach to each chunk
            
        Returns:
            List of Document objects
        """
        if metadata is None:
            metadata = {}
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Chunk by words
        chunks = self.chunk_text_by_words(cleaned_text)
        
        # Create Document objects
        documents = []
        for idx, chunk in enumerate(chunks):
            doc_metadata = {
                **metadata,
                "chunk_number": idx + 1,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks),
                "exact_data": chunk
            }
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        logger.info(f"Created {len(documents)} chunks from {len(cleaned_text)} characters")
        return documents
    
    def chunk_pages(self,
                   page_texts: List[str],
                   file_id: str,
                   file_name: str,
                   file_path: str) -> List[Document]:
        """
        Chunk multiple pages (e.g., from PDF)
        
        Args:
            page_texts: List of text from each page
            file_id: Unique file identifier
            file_name: Original filename
            file_path: File path
            
        Returns:
            List of Document objects from all pages
        """
        all_documents = []
        
        for page_number, text in enumerate(page_texts, start=1):
            chunks = self.chunk_text_by_words(text)
            
            for idx, chunk in enumerate(chunks):
                cleaned = self.preprocess_text(chunk)
                metadata = {
                    "chunk_number": idx + 1,
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "page_number": page_number,
                    "exact_data": chunk
                }
                all_documents.append(Document(
                    page_content=cleaned,
                    metadata=metadata
                ))
        
        logger.info(f"Created {len(all_documents)} document chunks from {len(page_texts)} pages")
        return all_documents


if __name__ == "__main__":
    # Test chunker
    chunker = TextChunker(word_limit=150, min_chunk_length=20)
    
    sample_text = """
    This is a sample document for testing the text chunker.
    It contains multiple paragraphs and sentences.
    The chunker will split this text into smaller pieces while maintaining context.
    This is useful for RAG systems where we need to retrieve relevant portions of documents.
    Each chunk will have metadata attached to it for tracking purposes.
    """ * 10  # Repeat to make it longer
    
    chunks = chunker.chunk_text(sample_text, metadata={"source": "test.txt"})
    
    print(f"\nCreated {len(chunks)} chunks")
    print(f"First chunk: {chunks[0].page_content[:100]}...")
    print(f"Metadata: {chunks[0].metadata}")
