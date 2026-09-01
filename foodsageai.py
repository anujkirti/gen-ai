import json
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

load_dotenv()

# 1. Define the schema
class Restaurant(BaseModel):
    name: str = Field(description="Restaurant name or 'Unknown'")
    cuisine: List[str] = Field(description="List of cuisines")
    location: str = Field(description="City/area or 'Unknown'")
    price_range: Optional[str] = Field(default="Not mentioned", description="Price range if mentioned, e.g. $, $$, $$$")
    popular_dishes: List[str] = Field(description="List of standout or popular dishes")
    rating: Optional[str] = Field(default="Not mentioned", description="Rating if mentioned")
    sentiment: Optional[str] = Field(description="Positive, Negative, Mixed, or Neutral")
    summary: str = Field(description="2-3 sentence original summary")
    key_points: List[str] = Field(description="Key takeaways or bullet points")

# 2. Use native structured output
model = ChatMistralAI(model="mistral-small-2603")
structured_model = model.with_structured_output(Restaurant)

# 3. Clean system prompt without conflicting formatting rules
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Food Sage, an AI that extracts structured metadata and summaries from restaurant/food text."),
    ("human", "{raw_text}")
])

chain = prompt | structured_model

para = input("Give your para about the restaurant or dish: ")
restaurant: Restaurant = chain.invoke({"raw_text": para})

# 4. Output as JSON
# Pydantic v2:
print(restaurant.model_dump_json(indent=4))

# If using Pydantic v1:
# print(restaurant.json(indent=4))
