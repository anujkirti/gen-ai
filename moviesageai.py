import json
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

load_dotenv()

# 1. Define the schema
class Movie(BaseModel):
    title: str = Field(description="Movie title or 'Unknown'")
    genre: List[str] = Field(description="List of genres")
    director: str = Field(description="Director name or 'Unknown'")
    cast: List[str] = Field(description="List of actors")
    release_year: Optional[int] = Field(default=None, description="Release year if available")
    rating: Optional[str] = Field(default="Not mentioned", description="Rating if mentioned")
    sentiment: Optional[str] = Field(description="Positive, Negative, Mixed, or Neutral")
    summary: str = Field(description="2-3 sentence original summary")
    key_points: List[str] = Field(description="Key takeaways or bullet points")

# 2. Use native structured output
model = ChatMistralAI(model="mistral-small-2603")
structured_model = model.with_structured_output(Movie)

# 3. Clean system prompt without conflicting formatting rules
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Movie Sage, an AI that extracts structured metadata and summaries from movie text."),
    ("human", "{raw_text}")
])

chain = prompt | structured_model

para = input("Give your para about the movie: ")
movie: Movie = chain.invoke({"raw_text": para})

# 4. Output as JSON
# Pydantic v2:
print(movie.model_dump_json(indent=4))

# If using Pydantic v1:
# print(movie.json(indent=4))