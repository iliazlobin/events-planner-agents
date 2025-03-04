import asyncio
import base64
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
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field

from events_agent.agents.web_surfer import MultimodalWebSurfer
from events_agent.assistant.default import CompleteOrEscalate
from events_agent.domain.state import State
from events_agent.utils.lang import create_tool_node_with_fallback, get_llm, print_message

config_list = [{"model": os.environ["OPENAI_API_MODEL_NAME"], "api_key": os.environ["OPENAI_API_KEY"]}]

llm_config = {
    "timeout": 60,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": ".web",
        "use_docker": False,
    },
    llm_config=llm_config,
    system_message="Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.",
)

# assistant_agent = autogen.AssistantAgent(
#     name="assistant_agent",
#     llm_config=llm_config,
# )


# async def create_web_surfer():
#     return MultimodalWebSurfer(
#         name="web_surfer_agent",
#         model_client=OpenAIChatCompletionClient(model=os.environ["OPENAI_API_MODEL_NAME"]),
#         start_page="https://google.com/",
#         description="I am a web surfer agent. I can help you with web browsing tasks. Always use Google for login, the browser should have a session available with creds.",
#         debug_dir=".web/debug",
#         # browser_data_dir=".web/browser_data",
#         downloads_folder=".web/downloads",
#         # use_ocr=True,
#         to_save_screenshots=True,
#         headless=False,
#         connect_over_cdp="http://192.168.1.33:9222",
#         # browser_data_dir="/mnt/c/Users/izlobin/chrome-debug",
#     )


web_surfer_agent = MultimodalWebSurfer(
    name="web_surfer_agent",
    model_client=OpenAIChatCompletionClient(model=os.environ["OPENAI_API_MODEL_NAME"]),
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


# async def do_web_action_without_picture(state: MessagesState):
#     tool_call = state["messages"][-1].tool_calls[0]
#     assert tool_call["name"] == "ToWebAction", "Expected tool call to be 'ToWebAction'"

#     request = tool_call["args"]["request"]
#     url = tool_call["args"]["url"]
#     task = f"Please perform the following task: {request}. The URL to visit is: {url}"

#     response = await web_surfer_agent.run(task=task)
#     last_message = response.messages[-1]
#     if last_message.type == "MultiModalMessage":
#         content = (
#             last_message.content[-2]
#             if isinstance(last_message.content, list) and isinstance(last_message.content[-1], autogen_core._image.Image)
#             else last_message.content[-1]
#         )
#     else:
#         content = last_message.content
#     return {
#         "messages": [
#             {
#                 "role": "tool",
#                 "tool_call_id": tool_call["id"],
#                 "content": content,
#             },
#         ],
#     }


async def do_web_action(state: State):
    tool_call = state["messages"][-1].tool_calls[0]
    assert tool_call["name"] == "ToWebAction", "Expected tool call to be 'ToWebAction'"

    request = tool_call["args"]["request"]
    url = tool_call["args"]["url"]
    task = f"Please perform the following task: {request}. The URL to visit is: {url}"

    response = await web_surfer_agent.run(task=task)
    last_message = response.messages[-1]
    if last_message.type == "MultiModalMessage":
        if isinstance(last_message.content, list) and isinstance(last_message.content[-1], autogen_core._image.Image):
            image = last_message.content[-1]
            tool_content = last_message.content[-2]
            content = [
                {
                    "type": "text",
                    "text": tool_content,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image.data_uri},
                },
            ]
        else:
            tool_content = last_message.content[-1]
    else:
        tool_content = last_message.content
    user_content = content if content else tool_content
    return {
        "messages": [
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_content,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    }


@tool
def pause_web_action() -> list[dict]:
    """Pause the web action and wait for user input."""

    return ""


class ToWebSupervisor(BaseModel):
    """Transfers work to a specialized assistant to handle web browsing tasks."""

    request: str = Field(
        description="A detailed task that the web supervisor agent should perform, including any necessary followup questions the web surfer agent should clarify before proceeding."
    )


class ToWebAction(BaseModel):
    """Transfers work to a specialized assistant to handle web browsing tasks."""

    request: str = Field(description="Any necessary followup questions the web surfer agent should clarify before proceeding.")
    url: str = Field(description="The URL of the web page to be visited.")


class CompleteWebAction(BaseModel):
    """
    CompleteWebSurfer is a model used to signal when the web surfer has completed its task.
    It can also indicate whether control of the dialog should be escalated to the main assistant
    for further actions based on the user's needs.

    Attributes:
        cancel (bool): A flag indicating whether the current task is canceled or completed.
                       Defaults to True.
        reason (str): A description providing the reason for the task's completion or the need
                      for further actions.

    Example:
            "reason": "I have fully completed the task."
            "reason": "I need to search the user's emails or calendar for more information."
    """

    cancel: bool = True
    reason: str

    class Config:
        schema_extra = {
            "examples": [
                {
                    "cancel": True,
                    "reason": "I have fully completed the task.",
                },
                {
                    "cancel": False,
                    "reason": "I need to search the user's emails or calendar for more information.",
                },
            ]
        }


class PauseWebAction(BaseModel):
    """
    PauseWebAction is a model used to signal when the web surfer needs to pause its task and wait for user input.

    Attributes:
        reason (str): A description providing the reason for the task's pause or the need
                      for further actions.

    Example:
            "reason": "I need the user to log in."
            "reason": "I need more information from the user."
    """

    reason: str

    class Config:
        schema_extra = {
            "examples": [
                {
                    "reason": "I need the user to log in.",
                },
                {
                    "reason": "I need more information from the user.",
                },
            ]
        }


web_supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Web Supervisor responsible for registering the user for events. "
            "Your sole purpose is to oversee the web surfer agent which has the capabilities to sign up / register / register again for events. "
            "You will fill out forms and complete event registrations. "
            "Look for confirmation messages to ensure the registration is successful. "
            "Check for buttons like 'register', 'sign up', 'register again' and proceed with the registration process. "
            "If the user is not logged in, interrupt and allow the user to log in. "
            "Assume the user is looged in, only interrupt if you see on the page that the user is not logged in. "
            "\n Current time: {time}."
            "\n User info: {user_info}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)


web_supervisor_runnable = web_supervisor_prompt | get_llm().bind_tools([ToWebAction, CompleteWebAction])
# web_supervisor_runnable = web_supervisor_prompt | get_llm().bind_tools([ToWebAction, PauseWebAction, CompleteWebAction])


# async def main() -> None:
#     graph = StateGraph(State)

#     builder.add_node("primary_assistant", Assistant(assistant_runnable))
#     builder.add_node("primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools))

#     graph.add_node(call_websurfer_agent)
#     graph.set_entry_point("call_autogen_agent")
#     graph = graph.compile()

#     thread_id = str(uuid.uuid4())

#     config = {
#         "configurable": {
#             "thread_id": thread_id,
#         }
#     }

#     tutorial_questions = ["Sign up for the meetup event (https://www.meetup.com/new-york-finance-and-banking-meetup/events/305160740)"]

#     printed = set()
#     for question in tutorial_questions:
#         async for message in graph.astream({"messages": ("user", question)}, config, stream_mode="values"):
#             print_message(message, printed)


# if __name__ == "__main__":
#     asyncio.run(main())
