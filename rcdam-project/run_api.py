"""Entry point to run the RC-DAM FastAPI server with Uvicorn."""
import os
import sys
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting RC-DAM Decision API on http://{host}:{port}")
    print(f"OpenAPI Documentation: http://{host}:{port}/docs")
    uvicorn.run("rcdam.api:app", host=host, port=port, reload=False)

