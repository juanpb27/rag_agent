from fastapi import FastAPI
from pydantic import BaseModel
from app.chat_engine import chat
from app.memory import clear_memory

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    user_input: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    result = await chat(request.session_id, request.user_input)
    return ChatResponse(response=result)

@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    clear_memory(session_id)
    return {"message": f"Memory cleared for session {session_id}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
