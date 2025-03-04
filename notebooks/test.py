import os
from typing import Any, Dict, List, Optional

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# Environment variables (match TypeScript)
OPENSEARCH_URL = os.environ["OPENSEARCH_URL"]  # e.g., "https://search-manual-test-....aos.us-east-1.on.aws"
ALL_EVENTS_INDEX = os.environ["ALL_EVENTS_INDEX"]  # e.g., "all-events"

# Singleton client instance
_client = None


def get_client() -> OpenSearch:
    """Get or create an OpenSearch client with AWS authentication."""
    global _client
    if _client:
        return _client

    # Use boto3's default credential chain (mimics TypeScript's defaultProvider)
    session = boto3.Session()
    credentials = session.get_credentials()
    if not credentials:
        raise ValueError("No AWS credentials found. Ensure AWS CLI is configured or use an IAM role.")

    aws_auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        "us-east-1",  # Match TypeScript region
        "es",  # Service name for OpenSearch
        session_token=credentials.token if credentials.token else None,
    )

    _client = OpenSearch(
        hosts=[{"host": OPENSEARCH_URL.replace("https://", ""), "port": 443}],
        http_auth=aws_auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

    # Test connection
    if not _client.ping():
        raise ConnectionError("Failed to connect to OpenSearch at " + OPENSEARCH_URL)

    return _client


def index_event(data: Dict[str, Any]) -> None:
    """Index a single event."""
    client = get_client()
    response = client.index(index=ALL_EVENTS_INDEX, id=data["url"], body=data)
    print(f"Event indexed: {data['url']}")


def get_all_events(size: int = 10, cursor: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve events with pagination, mirroring TypeScript's getAllEvents."""
    client = get_client()

    search_params = {
        "index": ALL_EVENTS_INDEX,
        "size": size,
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
                "description",
                "cover",
                "tags",
                "venue",
                "online",
                "eventSeriesDates",
                "price",
                "minPrice",
                "maxPrice",
                "spotsLeft",
                "waitlist",
                "registrationClosed",
                "non_commercial",
                "popularity",
                "free_admision",
                "no_additional_expenses",
                "drinks_provided",
                "food_provided",
                "venue_niceness",
                "quietness",
                "uniqueness",
                "proximity",
            ],
            "query": {"bool": {"must": [{"range": {"dateStart": {"gte": "now", "lte": "now+14d/d", "format": "strict_date_optional_time"}}}]}},
            "sort": [{"_score": "desc"}, {"_id": "asc"}],
        },
    }

    if cursor:
        search_params["body"]["search_after"] = cursor.split(",")

    try:
        response = client.search(**search_params)
        events = [hit["_source"] for hit in response["hits"]["hits"]]
        new_cursor = None
        if response["hits"]["hits"]:
            last_hit = response["hits"]["hits"][-1]
            if "sort" in last_hit:
                new_cursor = ",".join(map(str, last_hit["sort"]))

        print(f"Retrieved {len(events)} events")
        return {"events": events, "cursor": new_cursor}

    except Exception as e:
        print(f"Error retrieving events: {str(e)}")
        return {"events": [], "cursor": None}


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def book_event(url: str) -> dict:
    """
    Book an event by navigating to the provided URL and completing the booking process.

    Args:
        url (str): The URL of the event booking page.

    Returns:
        dict: A dictionary containing the booking confirmation details.
    """
    # Set up headless browser options
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Initialize the WebDriver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Open the event booking page
        driver.get(url)

        # Wait for the page to load and find the "Request to Join" button
        join_button = WebDriverWait(driver, 60).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'luma-button') and .//div[text()='Request to Join']]"))
        )
        join_button.click()

        # Fill out the form
        name_input = WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.NAME, "name")))
        name_input.send_keys("John Doe")

        email_input = driver.find_element(By.NAME, "email")
        email_input.send_keys("john.doe@example.com")

        phone_input = driver.find_element(By.NAME, "phone_number")
        phone_input.send_keys("+1 201 555 0123")

        company_input = driver.find_element(By.NAME, "registration_answers.0.answer")
        company_input.send_keys("Example Company")

        work_email_input = driver.find_element(By.NAME, "registration_answers.1.answer")
        work_email_input.send_keys("john.doe@company.com")

        linkedin_input = driver.find_element(By.NAME, "registration_answers.2.answer")
        linkedin_input.send_keys("https://linkedin.com/in/johndoe")
        # Explore the dropdown items available for company size
        company_size_input = driver.find_element(By.ID, ":r9:")
        company_size_input.click()
        company_size_options = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, "//div[@role='option']")))

        # Print out all available options
        for option in company_size_options:
            print(option.text)

        # Select the desired option
        desired_option_text = "51-200"
        for option in company_size_options:
            if option.text == desired_option_text:
            option.click()
            break

        persona_customer_input = driver.find_element(By.ID, ":rf:")
        persona_customer_input.click()
        persona_customer_option = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//div[text()='Yes']")))
        persona_customer_option.click()

        code_of_conduct_checkbox = driver.find_element(By.ID, ":rl:")
        code_of_conduct_checkbox.click()

        # Submit the booking form
        submit_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Confirm')]")))
        submit_button.click()

        # Wait for the confirmation message
        confirmation_message = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'confirmation-message')]")))

        # Return the confirmation details
        return {"status": "success", "message": confirmation_message.text}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        # Close the browser
        driver.quit()


def main():
    result = book_event("https://lu.ma/cryptomeetup")
    print(result)


if __name__ == "__main__":
    # Set environment variables if not already set (for testing)
    if "OPENSEARCH_URL" not in os.environ:
        os.environ["OPENSEARCH_URL"] = "search-manual-test-fczgibvrlzm6dobny7dhtzpqmq.aos.us-east-1.on.aws"
    if "ALL_EVENTS_INDEX" not in os.environ:
        os.environ["ALL_EVENTS_INDEX"] = "all-events"

    main()
