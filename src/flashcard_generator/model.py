from pydantic import BaseModel


class Flashcard(BaseModel):
    question: str
    answer: str


class FlashcardSet(BaseModel):
    flashcards: list[Flashcard]
