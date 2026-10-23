from fastapi import APIRouter

from src.services.conversation_store import conversations

router = APIRouter()


# Forget a conversation's history: its next message starts from scratch.
@router.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    conversations.clear(conversation_id)
    return {"message": f"Conversation '{conversation_id}' cleared."}
