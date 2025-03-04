import datetime
import json
import os
from typing import Annotated, Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

SCOPES = ["https://www.googleapis.com/auth/calendar"]
# SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def get_credentials():
    """Get Google API credentials."""
    creds = None
    if os.path.exists(".secrets/token.json"):
        creds = Credentials.from_authorized_user_file(".secrets/token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(".secrets/credentials.json", SCOPES)
            creds = flow.run_local_server(port=19000)
        with open(".secrets/token.json", "w") as token:
            token.write(creds.to_json())
    return creds


@tool
def get_calendar_events(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    start_time: Annotated[str, "Start time in ISO format in UTC"] = None,
    end_time: Annotated[str, "End time in ISO format in UTC"] = None,
) -> List[Dict]:
    """
    Retrieve all upcoming events from Google Calendar.

    Args:
        start_time (Annotated[str]): The start time of the range to view events in ISO format in UTC.
        end_time (Annotated[str]): The end time of the range to view events in ISO format in UTC.

    Returns:
        List[Dict]: A list of dictionaries containing the event details.
    """
    creds = get_credentials()

    service = build("calendar", "v3", credentials=creds)
    now = datetime.datetime.utcnow().isoformat() + "Z"  # 'Z' indicates UTC time
    time_min = start_time if start_time else now
    time_max = end_time if end_time else None

    events_result = service.events().list(calendarId="primary", timeMin=time_min, timeMax=time_max, singleEvents=True, orderBy="startTime").execute()
    events = events_result.get("items", [])
    calendar_events = []
    for event in events:
        description = event.get("description", "")
        first_line = description.split("\n")[0] if description else ""
        url = first_line if first_line.startswith("http://") or first_line.startswith("https://") else None

        calendar_event = {
            "url": url,
            "start": event.get("start"),
            "end": event.get("end"),
            "summary": event.get("summary"),
            "description": description,
            "location": event.get("location"),
            # "creator": event.get("creator"),
            # "organizer": event.get("organizer"),
        }
        calendar_events.append(calendar_event)

    events_status = state.get("events_status", {})
    for event in calendar_events:
        url = event.get("url")
        if url:
            url_status = events_status.get(url)
            if url_status:
                url_status["scheduled_to_calendar"] = True
                events_status[url] = url_status

    event_messages = [
        {
            "type": "text",
            "text": json.dumps(event),
        }
        for event in calendar_events
    ]

    return Command(
        update={
            "events_status": events_status,
            "messages": [ToolMessage(event_messages, tool_call_id=tool_call_id)],
        }
    )


@tool
def create_calendar_event(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    # events_status: Annotated[dict, InjectedState("events_status")],
    start_time: Annotated[str, "Event start time in ISO format in UTC"],
    end_time: Annotated[str, "Event end time in ISO format in UTC"],
    title: str,
    description: Optional[str] = None,
    url: Optional[str] = "",
    location: Optional[str] = None,
) -> Dict:
    """
    Create a new event in Google Calendar.

    Args:
        start_time (str): The start time of the event in ISO format in UTC.
        end_time (str): The end time of the event in ISO format in UTC.
        title (str): The title of the event.
        description (Optional[str]): A brief description of the event.
        url (Optional[str]): The URL of the event.
        location (Optional[str]): The location of the event.

    Returns:
        Dict: A dictionary containing the created event details.
    """
    creds = get_credentials()

    service = build("calendar", "v3", credentials=creds)
    url_description = f"{url}\n\n{description}" if url else description
    event = {
        "summary": title,
        "description": url_description,
        "location": location,
        "start": {"dateTime": start_time},
        "end": {"dateTime": end_time},
    }
    print(f"Creating calendar event: {event}")

    created_event = service.events().insert(calendarId="primary", body=event).execute()
    created_event_json = {
        "url": url,
        "start": created_event.get("start"),
        "end": created_event.get("end"),
        "summary": created_event.get("summary"),
        "description": created_event.get("description"),
        "location": created_event.get("location"),
    }
    print(f"Calendar event created: {created_event_json}")

    events_status = state.get("events_status", {})
    url_status = events_status.get(url, {})
    url_status["scheduled_to_calendar"] = True
    events_status[url] = url_status

    return Command(
        update={
            "events_status": events_status,
            "messages": [ToolMessage(created_event_json, tool_call_id=tool_call_id)],
        }
    )


if __name__ == "__main__":
    # # Prepare arguments for get_calendar_schedule
    # start_time = datetime.datetime.now()
    # end_time = start_time + datetime.timedelta(days=7)

    # # Call the tool using the invoke method with a dictionary of arguments
    # events = get_calendar_schedule.invoke({"start_time": start_time, "end_time": end_time})
    # print("Upcoming events:", events)

    # Prepare arguments for create_calendar_event
    start_time = datetime.datetime.now() + datetime.timedelta(days=1)
    end_time = datetime.datetime.now() + datetime.timedelta(days=1, hours=1)
    title = "Test Event"
    description = "This is a test event"
    location = "Virtual"

    # Call the tool using the invoke method with a dictionary of arguments
    new_event = create_calendar_event.invoke({"start_time": start_time, "end_time": end_time, "title": title, "description": description, "location": location})
    print("Created event:", new_event)
