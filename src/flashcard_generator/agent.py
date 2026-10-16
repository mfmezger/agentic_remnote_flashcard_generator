"""Gemini agent that turns a document into flashcards."""

import os

from google.genai.types import HarmBlockThreshold, HarmCategory
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

from flashcard_generator.model import FlashcardSet
from flashcard_generator.prompts import SYSTEM_PROMPT

UNBLOCKED_CATEGORIES = [
    HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    HarmCategory.HARM_CATEGORY_HARASSMENT,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
]


def initialize_agent(model_name: str = "gemini-2.5-flash") -> Agent[None, FlashcardSet]:
    """Build the flashcard agent; needs GEMINI_API_KEY in the environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY must be set (environment or .env)")

    model = GoogleModel(model_name=model_name, provider=GoogleProvider(api_key=api_key))
    settings = GoogleModelSettings(
        temperature=0.7,
        google_thinking_config={"thinking_budget": 5000},
        google_safety_settings=[
            {"category": category, "threshold": HarmBlockThreshold.BLOCK_NONE}
            for category in UNBLOCKED_CATEGORIES
        ],
    )
    return Agent(
        model=model,
        output_type=FlashcardSet,
        model_settings=settings,
        retries=5,
        system_prompt=SYSTEM_PROMPT,
    )
