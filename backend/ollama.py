import os
import asyncio

from llama_index.llms.ollama import Ollama
import yaml
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential

# Load configuration
file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)
global_config = yaml.safe_load(open(os.path.join(project_path, "config.yaml"), "r"))

OLLAMA_MODEL_NAME = global_config.get("backend").get("ollama_model_name", "qwen2:0.5b")

ollama_model = Ollama(model=OLLAMA_MODEL_NAME, request_timeout=300.0)

@retry(wait=wait_exponential(min=10, max=300), stop=stop_after_attempt(10))
async def query_ollama(prompt):
    """
    Query the Ollama model with a given prompt.

    Args:
        prompt (str): The input text to generate a title for.

    Returns:
        str: The generated title.
    """
    system_message = "You are a helpful assistant that provides concise titles and summaries."
    user_message = f"Please provide a concise title for the following text:\n\n{prompt}\n\nYou must return only the title."
    full_prompt = f"{system_message}\n {user_message}\n"
    return await asyncio.to_thread(ollama_model.complete, full_prompt).text