from datetime import date, datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, Field

from events_agent.domain.state import State
from events_agent.tools.events import search_events
from events_agent.utils.lang import get_llm


class EventsAssistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (not result.content or isinstance(result.content, list) and not result.content[0].get("text")):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class ToEventsAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle event bookings."""

    event_name: str = Field(description="The name of the event the user wants to book.")
    event_date: str = Field(description="The date of the event the user wants to book.")
    location: str = Field(description="The location of the event.")
    request: str = Field(description="Any additional information or requests from the user regarding the event booking.")

    class Config:
        json_schema_extra = {
            "example": {
                "event_name": "Concert",
                "event_date": "2023-09-10",
                "location": "New York",
                "request": "I would like front row seats if available.",
            }
        }


class CompleteEventsAssistant(BaseModel):
    """
    CompleteEventsAssistant is a model used to signal when the events assistant has completed its task.
    It can also indicate whether control of the dialog should be escalated to the main assistant
    for further actions based on the user's needs.

    Attributes:
        completed (bool): A flag indicating whether the current event-related task is completed.
                          Defaults to True.
        reason (str): A description providing the reason for the task's completion or the need
                      for further actions.

    Example:
            "reason": "I have successfully booked the event."
            "reason": "I need more information from the user to complete the booking."
    """

    completed: bool = True
    reason: str

    class Config:
        schema_extra = {
            "examples": [
                {
                    "completed": True,
                    "reason": "I have successfully booked the event.",
                },
                {
                    "completed": False,
                    "reason": "I need more information from the user to complete the booking.",
                },
            ]
        }


events_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for events planning. "
            "Your primary role is to search for event information and assist users with booking events. "
            "If a user requests to update or cancel an event booking, book a car rental, book a hotel, or get trip recommendations, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself. "
            "Only the specialized assistants are given permission to do this for the user. "
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the user, and always double-check the database before concluding that information is unavailable. "
            "When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If a search comes up empty, expand your search before giving up."
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)


events_assistant_safe_tools = [search_events]
# event_booking_sensitive_tools = [sign_up_for_event]
events_assistant_sensitive_tools = []
events_assistant_sensitive_tool_names = {t.name for t in events_assistant_sensitive_tools}
events_assistant_tools = events_assistant_safe_tools + events_assistant_sensitive_tools
events_assistant_runnable = events_assistant_prompt | get_llm().bind_tools(events_assistant_tools + [CompleteEventsAssistant])
# events_assistant_runnable = prompt | get_llm().bind_tools(events_assistant_tools + [ToEventsAssistant])
