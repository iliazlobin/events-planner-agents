from datetime import date, datetime
import json
from typing import Annotated, Any, Dict, Optional, List
from pydantic import BaseModel
from langgraph.prebuilt import InjectedState

from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command

from events_agent.client.opensearch import get_opensearch_client
from events_agent.domain.state import EventDetails, State
from events_agent.utils.lang import get_llm

# Constants
OPENSEARCH_URL = "https://search-manual-test-fczgibvrlzm6dobny7dhtzpqmq.aos.us-east-1.on.aws"
ALL_EVENTS_INDEX = "all-events"


@tool
def search_events(
    tool_call_id: Annotated[str, InjectedToolCallId],
    # events_status: Annotated[dict, InjectedState("events_status")],
    state: Annotated[dict, InjectedState],
    start_time: Optional[date | datetime] = None,
    end_time: Optional[date | datetime] = None,
    # limit: int = 10,
    # additional_fields: Optional[list[str]] = None,
):
    """Search for events based on the event time range."""

    # sort: Optional[list[Dict[str, Any]]] = (None,)

    client = get_opensearch_client()

    search_params = {
        "index": ALL_EVENTS_INDEX,
        "size": 5,
        "body": {
            "_source": [
                "url",
                "dateStart",
                "dateEnd",
                "title",
                "hosts",
                "group",
                "address",
                "guests",
                "attendees",
                "shortDescription",
                "cover",
                "tags",
                "venue",
                "online",
                "price",
                "spotsLeft",
            ],
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "dateStart": {
                                            "gte": start_time.isoformat() if start_time else "now",
                                            "lte": end_time.isoformat() if end_time else "now+14d/d",
                                            "format": "strict_date_optional_time",
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "script_score": {
                        "script": {
                            "source": """
                                double compositeScore = 0;
                                compositeScore += doc['popularity'].value * 0.1;
                                compositeScore += doc['uniqueness'].value * 0.2;
                                compositeScore += doc['venue_niceness'].value * 0.15;
                                compositeScore += doc['free_admision'].value * 0.2;
                                compositeScore += doc['drinks_provided'].value * 0.1;
                                compositeScore += doc['food_provided'].value * 0.1;
                                compositeScore += doc['quietness'].value * 0.05;
                                compositeScore += doc['proximity'].value * 0.05;
                                compositeScore += doc['non_commercial'].value * 0.025;
                                compositeScore += doc['no_additional_expenses'].value * 0.025;
                                
                                // Normalize composite score to be between 0 and 1
                                compositeScore = Math.min(Math.max(compositeScore, 0), 1);
                                
                                // Adjust the initial score based on the composite score
                                double adjustmentFactor = 1 + (compositeScore - 0.5) * 2;
                                return _score * adjustmentFactor;
                            """
                        }
                    },
                }
            },
            "sort": [{"_score": "desc"}, {"_id": "asc"}],
        },
    }

    response = client.search(**search_params)
    results = [EventDetails(**hit["_source"]) for hit in response["hits"]["hits"]]

    events_status = state.get("events_status", {})
    for result in results:
        url_status = events_status.get(result["url"], {})
        url_status["found"] = True
        url_status["registred"] = False
        url_status["scheduled_to_calendar"] = None
        url_status["details"] = result
        events_status[result["url"]] = url_status

    if results:
        print(f"Found {len(results)} events")
        print(f"First event: {results[0]}")

        results_json = [
            {
                "type": "text",
                "text": json.dumps(result),
            }
            for result in results
        ]

        return Command(
            update={
                "events_status": events_status,
                "messages": [ToolMessage(results_json, tool_call_id=tool_call_id)],
            }
        )
    return Command(
        update={
            "events_status": events_status,
            "messages": [ToolMessage("NOT_FOUND", tool_call_id=tool_call_id)],
        }
    )


@tool
def get_event_details(
    tool_call_id: Annotated[str, InjectedToolCallId],
    # events_status: Annotated[dict, InjectedState("events_status")],
    state: Annotated[dict, InjectedState],
    url: str,
):
    """Get event details based on the event URL."""
    client = get_opensearch_client()
    search_params = {
        "index": ALL_EVENTS_INDEX,
        "size": 1,
        "body": {
            "_source": [
                "url",
                "dateStart",
                "dateEnd",
                "title",
                "hosts",
                "group",
                "address",
                "guests",
                "attendees",
                "shortDescription",
                "cover",
                "tags",
                "venue",
                "online",
                "price",
                "spotsLeft",
            ],
            "query": {"term": {"url": url}},
        },
    }

    response = client.search(**search_params)
    results = [EventDetails(**hit["_source"]) for hit in response["hits"]["hits"]]
    result = results[0] if results else None

    events_status = state.get("events_status", {})
    url_status = events_status.get(url, {})
    if result:
        url_status["found"] = True
        url_status["details"] = result
    else:
        url_status["found"] = False
    events_status[url] = url_status

    if results:
        print(f"Found event: {results[0]}")
        result_json = json.dumps(results[0])
        return Command(
            update={
                "events_status": events_status,
                "messages": [ToolMessage(result_json, tool_call_id=tool_call_id)],
            }
        )
    return Command(
        update={
            "events_status": events_status,
            "messages": [ToolMessage("NOT_FOUND", tool_call_id=tool_call_id)],
        }
    )


safe_tools = [
    search_events,
    get_event_details,
]

sensitive_tools = []
sensitive_tool_names = {t.name for t in sensitive_tools}

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for an events planning application. Your primary role is to search for event information and assist users with booking events. "
            "If a user requests to update or cancel an event booking, delegate the task to the appropriate specialized assistant by invoking the corresponding tool. "
            "You are not able to make these types of changes yourself. Only the specialized assistants are given permission to do this for the user. "
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the user, and always double-check the database before concluding that information is unavailable. "
            "When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If a search comes up empty, expand your search before giving up."
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

assistant_runnable = assistant_prompt | get_llm().bind_tools(safe_tools + sensitive_tools)
