import asyncio
import uuid
from datetime import datetime
from typing import Callable, Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from regex import F

from events_agent.assistant.default import Assistant
from events_agent.assistant.events import CompleteEventsAssistant, EventsAssistant, ToEventsAssistant, events_assistant_runnable
from events_agent.assistant.primary import primary_assistant_runnable, primary_assistant_tools
from events_agent.domain.state import State
from events_agent.tools.calendar import create_calendar_event, get_calendar_events
from events_agent.tools.events import get_event_details, search_events
from events_agent.tools.user_info import fetch_user_info
from events_agent.tools.web_surfer import ToWebRegisterForEvent, web_register_for_event
from events_agent.utils.lang import create_tool_node_with_fallback, get_llm, print_message


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def user_info_action(state: State):
    return {
        "user_info": fetch_user_info.invoke({}),
        "events_status": {},
    }


async def create_graph():
    builder = StateGraph(State)

    supervisor_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an event concierge assisting the user with searching and registering for events. "
                "\n\nYou are leveraging the following tools:"
                "\n- Search for events: search_events"
                "\n- Get event details: get_event_details"
                "\n- Register for events: web_register_for_event"
                "\n- Check calendar events: get_calendar_events"
                "\n- Create calendar event: create_calendar_event"
                "\n\nYou follow the following algorithm:"
                "\n1. Search for events based on the user's request."
                "\n2. If the user provides a specific event URL, retrieve the event details."
                "\n3. If the user requests to register for an event, use the web_register_for_event tool."
                "\n4. If the user requests to create a calendar event, use the create_calendar_event tool after checking the calendar for this event to make sure it is not already there."
                "\n\nYou need to make sure that the user is registered for the event and the event is added to the calendar by confirming events statuses."
                "\nIf the status field isn't present then the status is unknown."
                "\nTo check if the event has already been scheduled to the calendar, you can use the get_calendar_events tool."
                "\nIf a tool call fails, you need to retry the tool call once."
                "\n\n Current time: {time}."
                "\n User info: {user_info}."
                "\n Events status: {events_status}.",
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now)
    supervisor_runnable = supervisor_prompt | get_llm().bind_tools(
        [
            search_events,
            get_event_details,
            # run_web_task,
            ToWebRegisterForEvent,
            create_calendar_event,
            get_calendar_events,
        ],
        parallel_tool_calls=False,
    )
    builder.add_node("supervisor", Assistant(supervisor_runnable))

    def route_supervisor(state: State):
        route = tools_condition(state)
        if route == END:
            return END
        tool_calls = state["messages"][-1].tool_calls
        print(f"Supervisor tool calls: {tool_calls}")
        if tool_calls:
            if any(tc["name"] == search_events.name for tc in tool_calls):
                return "search_events"
            if any(tc["name"] == get_event_details.name for tc in tool_calls):
                return "get_event_details"
            if any(tc["name"] == ToWebRegisterForEvent.__name__ for tc in tool_calls):
                return "web_register_for_event"
            if any(tc["name"] == create_calendar_event.name for tc in tool_calls):
                return "create_calendar_event"
            if any(tc["name"] == get_calendar_events.name for tc in tool_calls):
                return "get_calendar_events"
        raise ValueError("Invalid route")

    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        [
            "search_events",
            "get_event_details",
            # "run_web_task",
            "web_register_for_event",
            "create_calendar_event",
            "get_calendar_events",
            END,
        ],
    )

    builder.add_node("get_event_details", ToolNode([get_event_details]).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error"))
    builder.add_edge("get_event_details", "supervisor")

    builder.add_node("search_events", ToolNode([search_events]).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error"))
    builder.add_edge("search_events", "supervisor")

    # builder.add_node("run_web_task", ToolNode([run_web_task]).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error"))
    # builder.add_edge("run_web_task", "supervisor")

    builder.add_node("web_register_for_event", web_register_for_event)
    builder.add_edge("web_register_for_event", "supervisor")

    builder.add_node("create_calendar_event", ToolNode([create_calendar_event]).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error"))
    builder.add_edge("create_calendar_event", "supervisor")

    builder.add_node("get_calendar_events", ToolNode([get_calendar_events]).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error"))
    builder.add_edge("get_calendar_events", "supervisor")

    # Start workflow
    builder.add_node("user_info_action", user_info_action)
    builder.add_edge("user_info_action", "supervisor")
    builder.add_edge(START, "user_info_action")

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

    graph = await create_graph()

    result = await graph.ainvoke(
        # input={"messages": [HumanMessage(content="Find best events for the next week and sign up (register) for all of them through the web browser")]},
        # input={"messages": [HumanMessage(content="Sign up for this event https://lu.ma/h3qpiaqg")]},  # without dropdown menus
        # input={"messages": [HumanMessage(content="Sign up for this event https://lu.ma/x58kfr7r")]},  # with dropdown menus
        # input={"messages": [HumanMessage(content="Sign up for this event and/or confirm participation (url=https://lu.ma/x58kfr7r")]}, # with dropdown menu, works now
        # input={"messages": [HumanMessage(content="Sign up for this event https://lu.ma/y2viikq6")]},  # waitlist, bloom...
        # input={"messages": [HumanMessage(content="Sign up for this event https://lu.ma/0p7ujtek")]},  # shopify walk, march 9, 12 PM one click registration
        # input={"messages": [HumanMessage(content="Sign up for this event https://lu.ma/txi8bg6t")]},  # immigrant happy hour
        # input={
        #     "messages": [HumanMessage(content="Sign up for this event https://www.meetup.com/frenchies/events/303804266/")]
        # },  # meetup simple registration with form
        # input={"messages": [HumanMessage(content="Sign up for 2 best events for Monday and Tuesday next week. ")]},
        input={"messages": [HumanMessage(content="Let's find an event for Friday and Saturday next week after 3:00 PM and sign up for them. ")]},
        config=config,
        interrupt_before=["pause_web_action"],
    )
    # print(f"Result: {result}")

    snapshot = graph.get_state(config)
    while snapshot.next:
        try:
            user_input = input("Do you approve of the above actions? Type 'y' to continue;" " otherwise, explain your requested changed.\n")
        except:
            user_input = "y"

        if user_input.strip() == "y":
            result = await graph.ainvoke(
                None,
                config,
            )
        else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
            result = await graph.ainvoke(
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
        snapshot = graph.get_state(config)

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
