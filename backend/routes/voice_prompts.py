"""Search the local voice-description catalogue; no network lookup is needed."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import VoiceDesignPromptResponse
from ..services.voice_prompts import list_prompts

router = APIRouter(prefix="/voice-prompts", tags=["voice-prompts"])


@router.get("", response_model=list[VoiceDesignPromptResponse])
def get_voice_prompts(q: str = Query("", max_length=200), category: str = Query("", max_length=100), db: Session = Depends(get_db)):
    return list_prompts(db, q=q, category=category)
