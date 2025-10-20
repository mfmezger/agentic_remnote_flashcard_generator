"""Initialization of Google LLMs."""

import os
import textwrap

from google.genai.types import HarmBlockThreshold, HarmCategory
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from dotenv import load_dotenv

from flashcard_generator.model import FlashcardSet

load_dotenv(override=True)


def initialize_agent(
    prompt: str,
    temperature: float = 0.7,
    thinking_budget: int = 5000,
    model_name: str = "gemini-2.5-flash",
    tools: list = [],
    retries: int = 5,
) -> Agent[str, FlashcardSet]:
    """Initialize a Pydantic AI agent with Google model.

    Args:
        prompt: System prompt for the agent
        output_model: Pydantic model for structured output
        temperature: Model temperature for randomness
        thinking_budget: Budget for thinking tokens
        model_name: Name of the Google model to use
        tools: List of tools to be used by the agent

    Returns:
        Configured Pydantic AI agent

    """

    # make sure the GEMINI_API_KEY is set in the environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY must be set in the environment")

    provider = GoogleProvider(api_key=api_key)
    model = GoogleModel(model_name=model_name, provider=provider)
    model_settings = GoogleModelSettings(
        temperature=temperature,
        google_thinking_config={"thinking_budget": thinking_budget},
        google_safety_settings=[
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            },
        ],
    )

    return Agent(
        model=model,
        output_type=FlashcardSet,
        instrument=True,
        model_settings=model_settings,
        retries=retries,
        system_prompt=textwrap.dedent(
            text=prompt,
        ),
        tools=tools,
    )
