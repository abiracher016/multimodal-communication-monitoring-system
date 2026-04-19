"""
AI-Powered Live Session Monitoring System
Entry point — starts the FastAPI server.
"""
import uvicorn
from backend.config import API_HOST, API_PORT

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Session Monitoring System")
    print("  Starting FastAPI server...")
    print(f"  API:  http://localhost:{API_PORT}")
    print(f"  Docs: http://localhost:{API_PORT}/docs")
    print("=" * 60)

    uvicorn.run(
        "backend.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
