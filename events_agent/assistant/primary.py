import asyncio
import os
import shutil
import uuid
from datetime import datetime

import autogen
import autogen_core
import playwright
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchResults
from langgraph.graph import MessagesState, StateGraph
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field

from events_agent.agents.web_surfer import MultimodalWebSurfer
from events_agent.assistant.default import CompleteOrEscalate
from events_agent.assistant.events import ToEventsAssistant
from events_agent.assistant.web_surfer import ToWebSupervisor
from events_agent.domain.state import State
from events_agent.utils.lang import create_tool_node_with_fallback, get_llm, print_message

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for event planning. "
            "Your primary role is to search for event information, book events, and provide event recommendations to answer customer queries. "
            "You use a dedicated EventsAssistant to search for events. "
            "You use a dedicated WebSupervisor to register for events. "
            "\nCurrent user info: {user_info}"
            "\nCurrent time: {time}."
            "\nLocation: New York, NY.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)


class ToPrimaryAssistant(BaseModel):
    """Transfers work back to the primary assistant to evaluate task completion."""

    reason: str = Field(description="Reason for transferring back to the primary assistant.")

    class Config:
        json_schema_extra = {
            "example": {
                "reason": "Requesting evaluation of whether the initial task has been fully accomplished and completed.",
            }
        }


primary_assistant_tools = [
    # TavilySearchResults(max_results=1),
    # GoogleSearchResults(api_wrapper=GoogleSearchAPIWrapper(), num_results=1),
]
primary_assistant_runnable = primary_assistant_prompt | get_llm().bind_tools(
    primary_assistant_tools
    + [
        ToEventsAssistant,
        ToWebSupervisor,
    ]
)
