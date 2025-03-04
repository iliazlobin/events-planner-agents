import shutil
import uuid

from events_agent.graph.single import create_graph
from events_agent.utils.lang import print_message


def main():
    print("Start run single events agent")
    graph = create_graph()

    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            # "passenger_id": "3442 587242",
            "thread_id": thread_id,
        }
    }

    tutorial_questions = [
        # "What events are happening this weekend?",
        # "What are the top events in the city this month?",
        "Can you find any free events for this week?",
    ]

    printed = set()
    for question in tutorial_questions:
        messages = graph.stream({"messages": ("user", question)}, config, stream_mode="values")
        for message in messages:
            print_message(message, printed)
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
