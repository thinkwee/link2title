import os
import asyncio

import google.generativeai as genai
import yaml
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential

# Load configuration
file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)
global_config = yaml.safe_load(open(os.path.join(project_path, "config.yaml"), "r"))

GOOGLE_API_KEY = global_config.get("backend").get("google_api_key")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini models
model_10 = genai.GenerativeModel('gemini-1.0-pro-latest', generation_config={"max_output_tokens": 2048})
model_15 = genai.GenerativeModel('gemini-1.5-pro-latest', generation_config={"max_output_tokens": 8192})

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(10))
async def query_gemini(q):
    """
    Query the Gemini 1.0 model with a given prompt.

    Args:
        q (str): The input text to generate a title for.

    Returns:
        str: The generated title.

    Raises:
        ValueError: If there's an error in the model response.
    """
    try:
        system_message = "You are a helpful assistant that provides concise titles and summaries."
        user_message = f"Please provide a concise title for the following text:\n\n{q}\n\nYou must return only the title."
        response = await asyncio.to_thread(model_10.generate_content, f"{system_message}\n\nHuman: {user_message}")
        return response.text
    except ValueError as e:
        if hasattr(e, 'response'):
            response = e.response
            response_text = ""
            for candidate in response.candidates:
                response_text += " ".join([part.text for part in candidate.content.parts])
            return response_text
        else:
            raise e

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(10))
async def query_gemini_15(q):
    """
    Query the Gemini 1.5 model with a given prompt.

    Args:
        q (str): The input text to generate a title for.

    Returns:
        str: The generated title.

    Raises:
        ValueError: If there's an error in the model response.
    """
    try:
        system_message = "You are a helpful assistant that provides concise titles and summaries."
        user_message = f"Please provide a concise title for the following text:\n\n{q}\n\nYou must return only the title."
        response = await asyncio.to_thread(model_15.generate_content, f"{system_message}\n\nHuman: {user_message}")
        return response.text
    except ValueError as e:
        if hasattr(e, 'response'):
            response = e.response
            response_text = ""
            for candidate in response.candidates:
                response_text += " ".join([part.text for part in candidate.content.parts])
            return response_text
        else:
            raise e
