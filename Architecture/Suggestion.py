from pydantic import BaseModel


class Suggestion(BaseModel):
    description: str
    action: int
