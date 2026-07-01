#!/usr/bin/env python3
"""
Structured output extraction demo using LangChain and Pydantic.

The script demonstrates:
- Two Pydantic models: PersonInfo and MeetingNotes.
- Structured output parsing with LangChain's PydanticOutputParser.
- Simple routing logic to choose the correct model based on input text.
- CLI that runs two hard‑coded examples (person and meeting) and prints the parsed objects.
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# 1. Pydantic models
# --------------------------------------------------------------------------- #
class PersonInfo(BaseModel):
    name: str = Field(description="Full name of the person.")
    age: Optional[int] = Field(
        default=None, description="Age of the person, if known."
    )
    profession: str = Field(description="Current profession or job title.")
    skills: List[str] = Field(description="List of professional skills.")


class MeetingNotes(BaseModel):
    date: str = Field(description="Date of the meeting.")
    participants: List[str] = Field(description="List of meeting participants.")
    topics: List[str] = Field(description="Discussion topics.")
    decisions: List[str] = Field(description="Decisions made during the meeting.")
    next_steps: List[str] = Field(description="Action items for next steps.")


# --------------------------------------------------------------------------- #
# 2. Helper functions
# --------------------------------------------------------------------------- #
def build_parser(model_cls: type[BaseModel]) -> PydanticOutputParser:
    """Create a PydanticOutputParser for the given model class."""
    return PydanticOutputParser(pydantic_object=model_cls)


def build_prompt(parser: PydanticOutputParser) -> PromptTemplate:
    """Create a PromptTemplate that includes the parser's format instructions."""
    template = (
        "Extract structured data from the following text:\n\n"
        "{input_text}\n\n"
        "{format_instructions}"
    )
    return PromptTemplate.from_template(
        template, partial_variables={"format_instructions": parser.get_format_instructions()}
    )


def build_chain(llm: ChatOpenAI, prompt: PromptTemplate, parser: PydanticOutputParser):
    """Assemble the chain: prompt -> llm -> parser."""
    return prompt | llm | parser


def is_meeting(text: str) -> bool:
    """
    Very simple heuristic to decide whether the input text describes a meeting.
    Looks for keywords that are typical for meeting notes.
    """
    keywords = {"meeting", "date", "participants", "topics", "decisions", "next steps"}
    lowered = text.lower()
    return any(word in lowered for word in keywords)


def summarize(obj: BaseModel) -> str:
    """Return a concise summary string for the parsed object."""
    if isinstance(obj, PersonInfo):
        return (
            f"Person: {obj.name}, "
            f"Age: {obj.age if obj.age is not None else 'N/A'}, "
            f"Profession: {obj.profession}, "
            f"Skills: {', '.join(obj.skills)}"
        )
    if isinstance(obj, MeetingNotes):
        return (
            f"Meeting on {obj.date} with {', '.join(obj.participants)}. "
            f"Topics: {', '.join(obj.topics)}. "
            f"Decisions: {', '.join(obj.decisions)}. "
            f"Next steps: {', '.join(obj.next_steps)}"
        )
    return str(obj)


# --------------------------------------------------------------------------- #
# 3. Main execution
# --------------------------------------------------------------------------- #
def main():
    # Load environment variables (expects BROJS_PAT_TOKEN)
    load_dotenv()

    # Configure the LLM
    llm = ChatOpenAI(
        base_url="https://llm.brojs.ru/v1",
        api_key=os.getenv("BROJS_PAT_TOKEN"),
        model="openai/gpt-oss-20b",
        temperature=0.1,
    )

    # Build parsers, prompts, and chains for both models
    person_parser = build_parser(PersonInfo)
    meeting_parser = build_parser(MeetingNotes)

    person_prompt = build_prompt(person_parser)
    meeting_prompt = build_prompt(meeting_parser)

    person_chain = build_chain(llm, person_prompt, person_parser)
    meeting_chain = build_chain(llm, meeting_prompt, meeting_parser)

    # Hard‑coded examples
    examples = [
        (
            "Анна, 28 лет, Python-разработчик. Навыки: FastAPI, Docker.",
            "person",
        ),
        (
            "Дата: 2024-07-01\nУчастники: Иван, Мария, Алексей\nТемы: Планирование проекта, Распределение задач\nРешения: Принято использовать Agile, назначены спринты\nСледующие шаги: Составить backlog, назначить ответственных",
            "meeting",
        ),
    ]

    for idx, (text, expected_type) in enumerate(examples, start=1):
        print(f"\n=== Example {idx} ({expected_type}) ===")
        print(f"Input text:\n{text}\n")

        # Routing logic
        if is_meeting(text):
            chain = meeting_chain
            model_name = "MeetingNotes"
        else:
            chain = person_chain
            model_name = "PersonInfo"

        try:
            parsed_obj = chain.invoke({"input_text": text})
            print(f"Parsed {model_name} object:")
            print(parsed_obj.model_dump(indent=2))
            print("\nSummary:")
            print(summarize(parsed_obj))
        except Exception as e:
            print(f"Error parsing input: {e}")


if __name__ == "__main__":
    main()
