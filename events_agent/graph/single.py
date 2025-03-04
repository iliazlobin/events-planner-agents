import shutil
import uuid
from datetime import datetime

from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from events_agent.assistant.events import (
    EventsAssistant,
    events_assistant_prompt,
)
from events_agent.assistant.transitions import CompleteOrEscalate
from events_agent.domain.state import State
from events_agent.tools.calendar import get_calendar_events
from events_agent.tools.events import search_events
from events_agent.utils.lang import create_tool_node_with_fallback, get_llm, print_message

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an Event Recommendation Assistant. "
            "Your primary role is to find and recommend appropriate events for the user. "
            "You have access to the Google Calendar to check the user's schedule and the capability to search for events. "
            "When recommending events, consider the user's preferences and availability. "
            "Provide detailed information about each event, including date, time, location, and any other relevant details. "
            "Always double-check the database before concluding that information is unavailable. "
            "Be persistent in your search. If the initial search returns no results, expand your query bounds before giving up."
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)


def create_graph():
    builder = StateGraph(State)

    events_assistant_safe_tools = [
        search_events,
        get_calendar_events,
    ]
    events_assistant_sensitive_tools = []
    events_assistant_sensitive_tool_names = {t.name for t in events_assistant_sensitive_tools}
    events_assistant_tools = events_assistant_safe_tools + events_assistant_sensitive_tools
    events_assistant_runnable = events_assistant_prompt | get_llm().bind_tools(events_assistant_tools + [CompleteOrEscalate])

    builder.add_node("assistant", EventsAssistant(events_assistant_runnable))
    builder.add_edge(START, "assistant")

    builder.add_node("safe_tools", create_tool_node_with_fallback(events_assistant_safe_tools))
    builder.add_node("sensitive_tools", create_tool_node_with_fallback(events_assistant_sensitive_tools))

    def route_tools(state: State):
        next_node = tools_condition(state)
        if next_node == END:
            return END
        ai_message = state["messages"][-1]
        first_tool_call = ai_message.tool_calls[0]
        if first_tool_call["name"] in events_assistant_sensitive_tool_names:
            return "sensitive_tools"
        return "safe_tools"

    builder.add_conditional_edges("assistant", route_tools, ["safe_tools", "sensitive_tools", END])
    builder.add_edge("safe_tools", "assistant")
    builder.add_edge("sensitive_tools", "assistant")

    memory = MemorySaver()

    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["sensitive_tools"],
    )

    return graph


def main():
    print("Start run single events agent")
    graph = create_graph()

    thread_id = str(uuid.uuid4())
    log_file_path = f".log/{datetime.now().strftime('%Y%m%d_%H%M')}_{thread_id}.txt"

    config = {
        "configurable": {
            # "passenger_id": "3442 587242",
            "thread_id": thread_id,
        }
    }

    tutorial_questions = [
        "Find best events for me for this week based on my calendar availability",
        # "What events are happening this weekend?",
        # "What are the top events in the city this month?",
        # "Can you find any free events for this week?",
    ]

    printed = set()
    for question in tutorial_questions:
        messages = graph.stream({"messages": ("user", question)}, config, stream_mode="values")
        for message in messages:
            print_message(message, printed, log_file_path=log_file_path)
        snapshot = graph.get_state(config)
        while snapshot.next:
            try:
                user_input = input("Do you approve of the above actions? Type 'y' to continue;" " otherwise, explain your requested changed.\n\n")
            except:
                user_input = "y"
            if user_input.strip() == "y":
                result = graph.invoke(
                    None,
                    config,
                )
                print(result)
            else:
                result = graph.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=message["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                            )
                        ]
                    },
                    config,
                )
                print(result)
            snapshot = graph.get_state(config)


if __name__ == "__main__":
    main()


# display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
