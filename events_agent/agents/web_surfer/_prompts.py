WEB_SURFER_TOOL_PROMPT_MM = """
{state_description}

Below is a screenshot of the page, with interactive elements highlighted in colored bounding boxes. Each box has a numeric ID label matching its color. Additional details about each labeled element are provided below:

{visible_targets}{other_targets_str}{focused_hint}

Please respond to my next request by choosing an appropriate tool from the following set, or by directly answering the question if possible:

{tool_names}

When selecting tools, consider the following guidelines:
    - When filling out a form, if you encounter an interactive dropdown menu, click to expand valid options, and handle one task at a time.
    - Avoid inputting text into fields unless you have attempted to click the dropdown menu first.
    - For tasks involving the CURRENT VIEWPORT, actions like clicking links, clicking buttons, expanding dropdown menus, inputting text, or hovering over elements might be most suitable.
    - For tasks involving content found elsewhere on the CURRENT WEBPAGE [{title}]({url}), actions like scrolling, summarization, or full-page Q&A might be more appropriate.

Termination conditions:
    - When request is satisfied, call the tool 'complete' providing the boolean status of successful completion and reasoning behind it
    - In case of an error, call the tool 'error' providing the error message and reasoning behind it
    
My request follows:
"""

WEB_SURFER_TOOL_PROMPT_TEXT = """
{state_description}

You have also identified the following interactive components:

{visible_targets}{other_targets_str}{focused_hint}

Please respond to my next request by choosing an appropriate tool from the following set, or by directly answering the question if possible:

{tool_names}

When selecting tools, consider the following guidelines:
    - When filling out a form, if you encounter an interactive dropdown menu, click to expand valid options, and handle one task at a time.
    - Avoid inputting text into fields unless you have attempted to click the dropdown menu first.
    - For tasks involving the CURRENT VIEWPORT, actions like clicking links, clicking buttons, expanding dropdown menus, inputting text, or hovering over elements might be most suitable.
    - For tasks involving content found elsewhere on the CURRENT WEBPAGE [{title}]({url}), actions like scrolling, summarization, or full-page Q&A might be more appropriate.

Termination conditions:
    - When request is satisfied, call the tool 'complete' providing the boolean status of successful completion and reasoning behind it
    - In case of an error, call the tool 'error' providing the error message and reasoning behind it

My request follows:
"""


WEB_SURFER_QA_SYSTEM_MESSAGE = """
You are a helpful assistant that can summarize long documents to answer question.
"""


def WEB_SURFER_QA_PROMPT(title: str, question: str | None = None) -> str:
    base_prompt = f"We are visiting the webpage '{title}'. Its full-text content are pasted below, along with a screenshot of the page's current viewport."
    if question is not None:
        return f"{base_prompt} Please summarize the webpage into one or two paragraphs with respect to '{question}':\n\n"
    else:
        return f"{base_prompt} Please summarize the webpage into one or two paragraphs:\n\n"
