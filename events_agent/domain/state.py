from datetime import datetime
from typing import Annotated, List, Literal, Optional, TypedDict, Union

from langgraph.graph.message import AnyMessage, add_messages


class UserInfo(TypedDict):
    name: str
    email: str
    twitter: str
    linkedin: str
    company: str
    role: str


class EventDetails(TypedDict):
    url: str
    dateStart: datetime
    dateEnd: datetime
    title: str
    hosts: List[str]
    group: str
    address: str
    guests: List[str]
    attendees: List[str]
    shortDescription: str
    cover: str
    tags: List[str]
    venue: str
    online: bool
    price: float
    spotsLeft: int


class EventStatus(TypedDict):
    url: str
    details: EventDetails
    found: bool
    registred: bool
    scheduled_to_calendar: bool


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: UserInfo
    events_status: dict[str, EventStatus]


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]
