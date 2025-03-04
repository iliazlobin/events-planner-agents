from datetime import date, datetime
from typing import Dict, List, Optional

import pytz
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


@tool
def fetch_user_info() -> Dict[str, str]:
    """Fetch the user's info.

    Returns:
        A dictionary containing the user's info.
    """
    return {
        "name": "Ilia Zlobin",
        "email": "iliazlobin91@gmail.com",
        "businessEmail": "ilia.zlobin@mastercard.com",
        "twitter": "https://x.com/iliazlobin",
        "linkedin": "https://linkedin.com/in/iliazlobin",
        "company": "Mastercard",
        "role": "Principal Engineer",
        # "phone": "+1-551-275-4034",
        "age": "34",
    }
