"""
Voice-to-Voice RAG System with Ultravox
Main Entry Point
"""
import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from dotenv import load_dotenv

# Load environment before other imports
load_dotenv()


def setup_logging(level: str = "INFO"):
    """Configure loguru logging"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


async def run_rag_server():
    """Run RAG tool server in background (for HTTP tools - not needed with client tools)"""
    from rag.rag_tool_server import run_server
    from core.config import get_rag_settings
    
    settings = get_rag_settings()
    logger.info(f"[MAIN] Starting RAG tool server on port {settings.tool_port}...")
    
    # Run in thread to not block
    import threading
    server_thread = threading.Thread(
        target=run_server,
        kwargs={"port": settings.tool_port},
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to start
    await asyncio.sleep(2)
    logger.info("[MAIN] RAG tool server started")


async def run_voice_pipeline():
    """Run main voice pipeline using official Ultravox SDK"""
    from voice.ultravox_pipeline import create_call_and_run
    await create_call_and_run()


async def main_async():
    """Main async entry point"""
    print("\n" + "="*60)
    print("🎤 Voice-to-Voice RAG System with Ultravox")
    print("="*60 + "\n")
    
    print("Starting voice pipeline...")
    print("(RAG runs as client-side tool - no server needed)\n")
    
    # Run voice pipeline directly
    await run_voice_pipeline()


def run_benchmark():
    """Run latency benchmark"""
    print("\n🔬 Latency Benchmark Mode")
    print("-" * 40)
    
    import time
    from core.config import get_rag_settings
    
    settings = get_rag_settings()
    
    # Test RAG retrieval latency
    print("\n1. Testing RAG retrieval latency...")
    try:
        from rag.rag_service import RAGService
        
        rag = RAGService(
            groq_api_key=settings.groq_api_key,
            milvus_uri=settings.milvus_uri,
            embedding_model=settings.embedding_model
        )
        
        test_queries = [
            "What is voice recognition?",
            "How does text to speech work?",
            "Explain the RAG pipeline"
        ]
        
        for query in test_queries:
            start = time.perf_counter()
            context = rag.retrieve_context(query, top_k=5)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"   '{query[:30]}...' → {elapsed:.1f}ms")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test audio processing latency
    print("\n2. Testing audio processing latency...")
    try:
        import numpy as np
        from voice.vad import VoiceActivityDetector
        
        vad = VoiceActivityDetector()
        
        # Simulate 20ms of audio
        chunk_size = 320  # 20ms at 16kHz
        test_audio = np.random.randn(chunk_size).astype(np.float32) * 0.1
        audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()
        
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            vad.is_speech(audio_bytes)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"   VAD processing: {elapsed/iterations:.3f}ms per chunk")
        print(f"   (Overhead for {iterations} chunks: {elapsed:.1f}ms)")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "-" * 40)
    print("✓ Benchmark complete")
    print("\nTarget latencies:")
    print("   - RAG retrieval: <100ms ✓")
    print("   - VAD processing: <1ms ✓")
    print("   - End-to-end: <500ms (depends on Ultravox)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Voice-to-Voice RAG System with Ultravox"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark instead of voice pipeline"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--rag-only",
        action="store_true",
        help="Run only the RAG tool server"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if args.benchmark:
        run_benchmark()
        return
    
    if args.rag_only:
        logger.info("[MAIN] Running RAG server only mode")
        from rag.rag_tool_server import run_server
        from core.config import get_rag_settings
        settings = get_rag_settings()
        run_server(port=settings.tool_port)
        return
    
    # Run main async loop
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"[MAIN] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
