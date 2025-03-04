import asyncio
import base64
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import autogen
import autogen_core
import playwright
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import EVENT_LOGGER_NAME
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field

from events_agent.agents.web_surfer import MultimodalWebSurfer
from events_agent.assistant.default import CompleteOrEscalate
from events_agent.domain.state import State
from events_agent.utils.lang import create_tool_node_with_fallback, get_llm, print_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(logging.FileHandler(f".web/log/{datetime.now().strftime('%Y%m%d_%H%M%S')}_autogen.txt"))
logger.setLevel(logging.INFO)


web_surfer_agent = MultimodalWebSurfer(
    name="web_surfer_agent",
    model_client=OpenAIChatCompletionClient(model=os.environ["OPENAI_API_MODEL_NAME"], api_key=os.environ["OPENAI_API_KEY"]),
    start_page="https://google.com/",
    description="I am a web surfer agent. I can help you with web browsing tasks. Always use Google for login, the browser should have a session available with creds.",
    debug_dir=".web/debug",
    # browser_data_dir=".web/browser_data",
    downloads_folder=".web/downloads",
    # use_ocr=True,
    to_save_screenshots=True,
    headless=False,
    connect_over_cdp="http://192.168.1.33:9222",
    # browser_data_dir="/mnt/c/Users/izlobin/chrome-debug",
)

# @tool
# async def run_web_task(url: str, task: str, user_info: str) -> Dict:
#     """
#     Perform a web task using a browser session.

#     Args:
#         url (str): The URL to navigate to.
#         task (str): The task to perform on the webpage.

#     Returns:
#         content (str): The content of the webpage after performing the task.
#     """
#     task = f"Please perform the following task: {task}"
#     task += f"\n\nURL: {url}."
#     task += f"\nUser info: {user_info}"

#     # response = await web_surfer_agent.run(task=prompt)

#     critic_agent = AssistantAgent(
#         "critic",
#         model_client=OpenAIChatCompletionClient(model=os.environ["OPENAI_API_MODEL_NAME"], api_key=os.environ["OPENAI_API_KEY"]),
#         system_message="Evaluate the result of the previous agent. Respond with 'COMPLETED' if the task is completed, otherwise provide feedback.",
#     )

#     agent_team = RoundRobinGroupChat([web_surfer_agent, critic_agent], max_turns=20, termination_condition=TextMentionTermination("COMPLETED"))
#     response = await agent_team.run(task=task)

#     last_message = response.messages[-1]
#     if last_message.type == "MultiModalMessage":
#         if isinstance(last_message.content, list) and isinstance(last_message.content[-1], autogen_core._image.Image):
#             content = last_message.content[-2]
#         else:
#             content = last_message.content[-1]
#     else:
#         content = last_message.content
#     return content


from typing import Sequence

from autogen_agentchat.base import TerminatedException, TerminationCondition
from autogen_agentchat.messages import AgentEvent, ChatMessage, StopMessage, ToolCallExecutionEvent
from autogen_core import Component
from pydantic import BaseModel
from typing_extensions import Self


def approve(reasoning: str) -> None:
    """Approve the message when all feedbacks have been addressed."""
    print(f"Approved. Reasoning: {reasoning}")
    pass


# class RunWebAction(BaseModel):
#     """Transfers work to a specialized assistant to handle web browsing tasks."""

#     request: str = Field(description="Any necessary followup questions the web surfer agent should clarify before proceeding.")
#     url: str = Field(description="The URL of the web page to be visited.")


class ToWebRegisterForEvent(BaseModel):
    """Transfers work to a specialized assistant to handle web browsing tasks."""

    request: str = Field(description="Additional information about the event registration request.")
    url: str = Field(description="The URL of the event page.")


async def web_register_for_event(state: State):
    tool_call = state["messages"][-1].tool_calls[0]

    assert tool_call["name"] == "ToWebRegisterForEvent", "Expected tool call to be 'ToWebRegisterForEvent'"

    request = tool_call["args"]["request"]
    url = tool_call["args"]["url"]
    user_info = state["user_info"]

    task = "Register the user for the event at the specified URL:"
    if request:
        task += f"\nRequest: {request}"

    task += f"\nURL: {url}."
    task += f"\nUser info: {user_info}"

    task += f"\n\nInstructions:"
    task += f"\nThe user can already be registred for the event, if so complete the procedure by calling the 'complete' tool."
    task += f"\nAssume the user is logged in to the website, if the aren't exit out with error by calling the 'error' tool."
    task += f"\nIf you see a form to fill out along with the registration button, fill out the form with the user info and then click the button."
    task += f"\nIf you see mandatory checkboxes (with an asterisk), click on them to complete the registration form."

    # response = await web_surfer_agent.run(task=prompt)

    # critic_agent = AssistantAgent(
    #     "critic",
    #     model_client=OpenAIChatCompletionClient(model=os.environ["OPENAI_API_MODEL_NAME"], api_key=os.environ["OPENAI_API_KEY"]),
    #     tools=[approve],
    #     system_message="Provide constructive feedback. Use the approve tool to approve when all feedbacks are addressed and the initial task is completed.",
    # )

    # function_call_termination = FunctionCallTermination(function_name="approve")
    # agent_team = RoundRobinGroupChat([web_surfer_agent], max_turns=2, termination_condition=web_surfer_agent.termination_condition)

    termination_condition = TextMentionTermination("COMPLETED") | TextMentionTermination("ERROR")
    agent_team = RoundRobinGroupChat([web_surfer_agent], max_turns=15, termination_condition=termination_condition)
    response = await agent_team.run(task=task)

    last_message = response.messages[-1]
    if last_message.type == "MultiModalMessage":
        if isinstance(last_message.content, list) and isinstance(last_message.content[-1], autogen_core._image.Image):
            content = last_message.content[-2]
        else:
            content = last_message.content[-1]
    else:
        content = last_message.content

    # DEBUG
    # content = "COMPLETED"

    registered = True if content == "COMPLETED" else False

    events_status = state.get("events_status", {})
    url_status = events_status.get(url, {})
    url_status["registered"] = registered
    events_status[url] = url_status

    return Command(
        update={
            "events_status": events_status,
            "messages": [ToolMessage(content, tool_call_id=tool_call["id"])],
        }
    )


async def main() -> None:
    # Define a team
    agent_team = RoundRobinGroupChat([web_surfer_agent], max_turns=20)

    # stream = agent_team.run_stream(task="Register for the Climate Tech Demo Night event: https://lu.ma/9blmqsnp")
    # await Console(stream)

    result = await agent_team.run(
        task='Register for the Climate Tech Demo Night event: https://lu.ma/9blmqsnp, user_info: \{"name": "Ilia Zlobin","email": "iliazlobin91@gmail.com","twitter": "https://x.com/iliazlobin","linkedin": "https://linkedin.com/in/iliazlobin","company": "Mastercard","role": "Principal Engineer",\}'
    )
    print(result)

    await web_surfer_agent.close()


if __name__ == "__main__":
    asyncio.run(main())
