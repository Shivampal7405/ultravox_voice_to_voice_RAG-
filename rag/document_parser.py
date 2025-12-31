"""
Apache Tika Document Parser
Extracts text from various document formats using Docker-based Tika server
"""
import requests
from typing import Optional
from pathlib import Path
import os


class TikaDocumentParser:
    """Document parser using Apache Tika"""
    
    def __init__(self, tika_url: str = "http://localhost:9998"):
        """
        Initialize Tika parser
        
        Args:
            tika_url: URL of Tika server (default: http://localhost:9998)
        """
        self.tika_url = tika_url
        self.tika_endpoint = f"{tika_url}/tika"
    
    def is_available(self) -> bool:
        """
        Check if Tika server is available
        
        Returns:
            True if Tika is running, False otherwise
        """
        try:
            response = requests.get(self.tika_url, timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def parse_document(self, file_path: str) -> Optional[str]:
        """
        Extract text from document using Tika
        
        Args:
            file_path: Path to document file
            
        Returns:
            Extracted text or None if failed
        """
        if not os.path.exists(file_path):
            print(f"[TIKA ERROR] File not found: {file_path}")
            return None
        
        try:
            print(f"[TIKA] Parsing document: {Path(file_path).name}")
            
            with open(file_path, 'rb') as f:
                response = requests.put(
                    self.tika_endpoint,
                    data=f,
                    headers={'Accept': 'text/plain'},
                    timeout=60
                )
            
            if response.ok:
                text = response.text.strip()
                print(f"[TIKA] Extracted {len(text)} characters")
                return text
            else:
                print(f"[TIKA ERROR] Failed to parse document: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[TIKA ERROR] Exception during parsing: {e}")
            return None
    
    def parse_bytes(self, file_bytes: bytes, filename: str = "document") -> Optional[str]:
        """
        Extract text from document bytes
        
        Args:
            file_bytes: Document bytes
            filename: Original filename (for logging)
            
        Returns:
            Extracted text or None if failed
        """
        try:
            print(f"[TIKA] Parsing document: {filename}")
            
            response = requests.put(
                self.tika_endpoint,
                data=file_bytes,
                headers={'Accept': 'text/plain'},
                timeout=60
            )
            
            if response.ok:
                text = response.text.strip()
                print(f"[TIKA] Extracted {len(text)} characters")
                return text
            else:
                print(f"[TIKA ERROR] Failed to parse document: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[TIKA ERROR] Exception during parsing: {e}")
            return None


if __name__ == "__main__":
    # Test Tika parser
    parser = TikaDocumentParser()
    
    if parser.is_available():
        print("✓ Tika server is available")
    else:
        print("✗ Tika server is NOT available")
        print("Start Tika with: docker run -d -p 9998:9998 apache/tika")
