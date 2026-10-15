import os
import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()
load_dotenv(Path(__file__).parent / ".env")

app = FastAPI(title="Sports Chatbot UI", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")

SYSTEM_PROMPT = (
    "You are Sportix AI, an elite, comprehensive sports domain expert chatbot. "
    "You possess deep, encyclopedic knowledge of all sports worldwide—including rules, "
    "history, athletes, statistics, tactics, tournaments, leagues, and records across "
    "football/soccer, basketball, cricket, Formula 1, tennis, rugby, baseball, athletics, "
    "swimming, MMA/boxing, winter sports, esports, and beyond.\n\n"
    "CRITICAL GUARDRAIL:\n"
    "You must STRICTLY answer questions only related to sports. If a user asks about ANY "
    "other domain or subject (such as coding, mathematics, politics, finance, geography, "
    "cooking, movies, general knowledge, etc.), you must politely decline and state:\n"
    "'I am Sportix AI, an AI dedicated exclusively to the sports domain. I can only assist "
    "with sports-related questions, rules, records, and athletes. Please ask me anything about sports!'"
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "gpt-4o-mini"
    api_key: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    model: str

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    key = req.api_key.strip() if req.api_key and req.api_key.strip() else DEFAULT_API_KEY
    if not key or key == "YOUR_OPENAI_API_KEY":
        raise HTTPException(
            status_code=400,
            detail="OpenAI API Key is required. Please set it in the UI or environment."
        )

    try:
        client = OpenAI(api_key=key)
        
        # Prepare messages payload ensuring the system guardrail is at the top
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in req.messages:
            if m.role in ["user", "assistant"]:
                api_messages.append({"role": m.role, "content": m.content})

        completion = client.chat.completions.create(
            model=req.model or "gpt-4o-mini",
            messages=api_messages,
            temperature=0.7
        )

        reply_content = completion.choices[0].message.content
        return ChatResponse(reply=reply_content, model=req.model or "gpt-4o-mini")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
def list_models_endpoint(api_key: Optional[str] = None):
    key = api_key.strip() if api_key and api_key.strip() else DEFAULT_API_KEY
    try:
        client = OpenAI(api_key=key)
        models = client.models.list()
        chat_models = [
            m.id for m in models.data 
            if any(p in m.id for p in ["gpt-4o", "gpt-4", "gpt-3.5", "o1", "o3"])
        ]
        return {"models": sorted(chat_models)}
    except Exception as e:
        return {"models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], "error": str(e)}

@app.get("/")
def get_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(index_file)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  🏆 SPORTIX AI - SPORTS CHATBOT WEB UI")
    print("  Server running at: http://127.0.0.1:8000")
    print("=" * 60 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
