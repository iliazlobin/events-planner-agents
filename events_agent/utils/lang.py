import os

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("OPENAI_API_MODEL_NAME", "gpt-4o-mini"))
    return _llm_instance


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


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")


def print_message(event: dict, _printed: set, max_length=1500, log_file_path=".log/default.txt"):
    current_state = event.get("dialog_state")
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if current_state:
                msg_repr = f"{msg_repr}\n"
                f"dialog_state{current_state[-1]}"
            if len(msg_repr) > max_length:
                print(msg_repr[:max_length] + " ... (truncated)")
            else:
                print(msg_repr)
            _printed.add(message.id)
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{msg_repr}\n")
