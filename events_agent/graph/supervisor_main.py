import asyncio
import uuid
from datetime import datetime
from typing import Callable, Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from events_agent.assistant.default import Assistant
from events_agent.assistant.events import CompleteEventsAssistant, EventsAssistant, ToEventsAssistant, events_assistant_runnable
from events_agent.assistant.primary import primary_assistant_runnable, primary_assistant_tools
from events_agent.assistant.web_surfer import (
    CompleteWebAction,
    PauseWebAction,
    ToWebAction,
    ToWebSupervisor,
    do_web_action,
    pause_web_action,
    web_supervisor_runnable,
)
from events_agent.domain.state import State
from events_agent.tools.events import search_events
from events_agent.tools.user_info import fetch_user_info
from events_agent.utils.lang import create_tool_node_with_fallback, print_message


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


def create_back_to_primary() -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content="Transitioning back to the primary assistant to evaluate whether the initial task has been accomplished or not. ",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": "primary_assistant",
        }

    return entry_node


def user_info(state: State):
    return {"user_info": fetch_user_info.invoke({})}


async def create_graph():
    builder = StateGraph(State)

    builder.add_node("fetch_user_info", user_info)
    builder.add_edge(START, "fetch_user_info")

    builder.add_node(
        "enter_web_supervisor",
        create_entry_node("Web Supervisor / Surfer (events registration)", "web_supervisor"),
    )
    builder.add_edge("enter_web_supervisor", "web_supervisor")

    # def pop_dialog_state(state: State) -> dict:
    #     """Pop the dialog stack and return to the main assistant.

    #     This lets the full graph explicitly track the dialog flow and delegate control
    #     to specific sub-graphs.
    #     """
    #     messages = []
    #     if state["messages"][-1].tool_calls:
    #         # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
    #         messages.append(
    #             ToolMessage(
    #                 content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
    #                 tool_call_id=state["messages"][-1].tool_calls[0]["id"],
    #             )
    #         )
    #     return {
    #         "dialog_state": "pop",
    #         "messages": messages,
    #     }

    # builder.add_node("leave_skill", pop_dialog_state)
    # builder.add_edge("leave_skill", "primary_assistant")

    builder.add_node("back_to_primary", create_back_to_primary())
    builder.add_edge("back_to_primary", "primary_assistant")

    builder.add_node("web_supervisor", Assistant(web_supervisor_runnable))

    # def route_web_supervisor(
    #     state: State,
    # ):
    #     route = tools_condition(state)
    #     if route == END:
    #         return END
    #     tool_calls = state["messages"][-1].tool_calls
    #     if tool_calls:
    #         if tool_calls[0]["name"] == ToWebAction.__name__:
    #             return "web_surfer"
    #         if tool_calls[0]["name"] == CompleteWebAction.__name__:
    #             return "web_supervisor"
    #     raise ValueError("Invalid route")
    #     # return "web_supervisor"

    def route_web_supervisor(
        state: State,
    ):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        did_cancel = any(tc["name"] == CompleteWebAction.__name__ for tc in tool_calls)
        if did_cancel:
            return "back_to_primary"
        if all(tc["name"] in [ToWebAction.__name__] for tc in tool_calls):
            return "web_action"
        # if all(tc["name"] in [PauseWebAction.__name__] for tc in tool_calls):
        #     return "pause_web_action"
        raise ValueError("Invalid route")

    builder.add_conditional_edges(
        "web_supervisor",
        route_web_supervisor,
        [
            "web_action",
            # "pause_web_action",
            # "web_supervisor_tools",
            "back_to_primary",
            END,
        ],
    )
    # builder.add_edge("web_supervisor_tools", "web_supervisor")

    # builder.add_node(
    #     "enter_web_surfer",
    #     create_entry_node("Web Surfer", "web_surfer"),
    # )
    builder.add_node(
        "web_action",
        do_web_action,
    )
    builder.add_edge("web_action", "web_supervisor")

    # builder.add_node(
    #     "pause_web_action",
    #     pause_web_action,
    # )
    # builder.add_edge("pause_web_action", "web_supervisor")

    # Events Assistant (search)
    events_assistant_tools = [
        search_events,
    ]
    builder.add_node("events_assistant", EventsAssistant(events_assistant_runnable))
    builder.add_node("events_assistant_tools", create_tool_node_with_fallback(events_assistant_tools))

    builder.add_node(
        "enter_events_assistant",
        create_entry_node("Events Assistant (search)", "events_assistant"),
    )
    builder.add_edge("enter_events_assistant", "events_assistant")

    def route_events_assistant(
        state: State,
    ):
        # route = tools_condition(state)
        # if route == END:
        #     return END
        tool_calls = state["messages"][-1].tool_calls
        did_cancel = any(tc["name"] == CompleteEventsAssistant.__name__ for tc in tool_calls)
        if did_cancel:
            return "back_to_primary"
        toolnames = [t.name for t in events_assistant_tools]
        if all(tc["name"] in toolnames for tc in tool_calls):
            return "events_assistant_tools"
        raise ValueError("Invalid route")

    builder.add_conditional_edges(
        "events_assistant",
        route_events_assistant,
        [
            "events_assistant_tools",
            "back_to_primary",
            END,
        ],
    )
    builder.add_edge("events_assistant_tools", "events_assistant")

    # Primary Assistant
    builder.add_node("primary_assistant", Assistant(primary_assistant_runnable))
    builder.add_node("primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools))

    def route_primary_assistant(
        state: State,
    ):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        if tool_calls:
            if tool_calls[0]["name"] == ToWebSupervisor.__name__:
                return "enter_web_supervisor"
            if tool_calls[0]["name"] == ToEventsAssistant.__name__:
                return "enter_events_assistant"
            return "primary_assistant_tools"
        raise ValueError("Invalid route")

    builder.add_conditional_edges(
        "primary_assistant",
        route_primary_assistant,
        [
            "enter_web_supervisor",
            "enter_events_assistant",
            "primary_assistant_tools",
            END,
        ],
    )
    builder.add_edge("primary_assistant_tools", "primary_assistant")

    def route_to_workflow(
        state: State,
    ) -> Literal[
        "primary_assistant",
        "events_assistant",
        "web_supervisor",
    ]:
        """If we are in a delegated state, route directly to the appropriate assistant."""
        dialog_state = state.get("dialog_state")
        if not dialog_state:
            return "primary_assistant"
        return dialog_state[-1]

    builder.add_conditional_edges("fetch_user_info", route_to_workflow)

    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=[
            # "web_supervisor_sensitive_tools",
        ],
    )

    return graph


async def main():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    # log_file_path = f".log/{datetime.now().strftime('%Y%m%d_%H%M')}_{thread_id}.txt"

    app = await create_graph()

    result = await app.ainvoke(
        # input={"messages": [HumanMessage(content="Register for the event: https://www.meetup.com/new-york-finance-and-banking-meetup/events/305160740")]},
        # input={"messages": [HumanMessage(content="Register for the event: https://lu.ma/bfftji1b?tk=tlMBTh")]},
        # input={"messages": [HumanMessage(content="Find one good event for the weekend and register for it")]},
        input={"messages": [HumanMessage(content="Find best events for the next week and sign up (register) for all of them through the web browser")]},
        config=config,
        interrupt_before=["pause_web_action"],
    )
    # print(f"Result: {result}")

    snapshot = app.get_state(config)
    while snapshot.next:
        try:
            user_input = input("Do you approve of the above actions? Type 'y' to continue;" " otherwise, explain your requested changed.\n")
        except:
            user_input = "y"

        if user_input.strip() == "y":
            result = await app.ainvoke(
                None,
                config,
            )
        else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
            result = await app.ainvoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=result["messages"][-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
                config,
            )
        snapshot = app.get_state(config)

    # config = {
    #     "configurable": {
    #         "thread_id": thread_id,
    #     }
    # }

    # tasks = [
    #     # "Sign up for the meetup event (https://www.meetup.com/new-york-finance-and-banking-meetup/events/305160740)",
    #     "Register for the event (https://lu.ma/bfftji1b?tk=tlMBTh)",
    # ]

    # printed = set()
    # for task in tasks:
    #     async for message in app.astream({"messages": ("user", task)}, config, stream_mode="values"):
    #         print_message(message, printed, log_file_path=log_file_path)

    # printed = set()
    # for task in tasks:
    #     messages = graph.astream({"messages": ("user", task)}, config, stream_mode="values")
    #     for message in messages:
    #         print_message(message, printed, log_file_path=log_file_path)
    #     snapshot = graph.get_state(config)
    #     while snapshot.next:
    #         try:
    #             user_input = input("Do you approve of the above actions? Type 'y' to continue; otherwise, explain your requested changes.\n\n")
    #         except:
    #             user_input = "y"
    #         if user_input.strip() == "y":
    #             result = graph.invoke(
    #                 None,
    #                 config,
    #             )
    #             print(result)
    #         else:
    #             result = graph.invoke(
    #                 {
    #                     "messages": [
    #                         ToolMessage(
    #                             tool_call_id=message["messages"][-1].tool_calls[0]["id"],
    #                             content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
    #                         )
    #                     ]
    #                 },
    #                 config,
    #             )
    #             print(result)
    #         snapshot = graph.get_state(config)


if __name__ == "__main__":
    asyncio.run(main())
