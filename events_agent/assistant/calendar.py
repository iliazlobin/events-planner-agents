import datetime
import os

from langchain_community.tools.google_calendar import GoogleCalendarCreateTool, GoogleCalendarViewTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, Field

from events_agent.utils.lang import get_llm

calendar_assistant_params = {
    "credentials": {
        "clientEmail": os.getenv("GOOGLE_CALENDAR_CLIENT_EMAIL"),
        "privateKey": os.getenv("GOOGLE_CALENDAR_PRIVATE_KEY"),
        "calendarId": os.getenv("GOOGLE_CALENDAR_CALENDAR_ID"),
    },
    "scopes": [
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/calendar.events",
    ],
    "model": get_llm(),
}

# calendar_assistant_create_tool = GoogleCalendarCreateTool(calendar_assistant_params)
calendar_assistant_view_tool = GoogleCalendarViewTool(calendar_assistant_params)

calendar_assistant_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for managing calendar availability. "
            "The primary assistant delegates work to you whenever the user needs help identifying gaps in their calendar. "
            "Your sole purpose is to identify availability gaps in the user's calendar. "
            "When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            "Remember that a task isn't completed until after the relevant tool has successfully been used."
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

calendar_assistant_safe_tools = [calendar_assistant_view_tool]
calendar_assistant_sensitive_tools = []
calendar_assistant_tools = calendar_assistant_safe_tools + calendar_assistant_sensitive_tools
calendar_assistant_runnable = calendar_assistant_assistant_prompt | get_llm().bind_tools(calendar_assistant_tools)


class ToCalendarAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle calendar management."""

    task: str = Field(description="The task the user wants to perform on their calendar.")
    details: str = Field(description="Any additional information or requests from the user regarding the calendar task.")

    class Config:
        json_schema_extra = {
            "example": {
                "task": "Identify gaps",
                "details": "Find gaps in my calendar for the next week.",
            }
        }
