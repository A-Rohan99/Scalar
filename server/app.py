import os
import uvicorn

# Re-export the FastAPI app so the validator/tests can import it.
from app.api import app

def main():
    """Fallback entrypoint if run directly."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app.api:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
