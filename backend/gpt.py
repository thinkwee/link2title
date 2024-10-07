import os
import traceback
from typing import Dict
import asyncio

import openai
import tiktoken
import yaml
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential

# Load configuration
file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)
global_config = yaml.safe_load(open(os.path.join(project_path, "config.yaml"), "r"))

OPENAI_API_KEY = global_config.get("backend").get("openai_api_key")
BASE_URL = global_config.get("backend").get("base_url", None)

# Initialize OpenAI client
if BASE_URL:
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL,
    )
else:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)


def calc_max_token(messages, model):
    """
    Calculate the maximum number of tokens for a given model and messages.

    Args:
        messages (list): List of message dictionaries.
        model (str): The name of the model.

    Returns:
        int: The maximum number of completion tokens allowed.
    """
    string = "\n".join([message["content"] for message in messages])
    encoding = tiktoken.encoding_for_model(model)
    num_prompt_tokens = len(encoding.encode(string))
    gap_between_send_receive = 15 * len(messages)
    num_prompt_tokens += gap_between_send_receive

    num_max_token_map = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo-0613": 4096,
        "gpt-3.5-turbo-16k-0613": 16384,
        "gpt-4": 8192,
        "gpt-4-0125-preview": 128000,
        "gpt-4-turbo": 128000,
        "claude-3-sonnet-20240229": 200000,
        "gpt-4o-mini": 128000
    }
    num_max_token = num_max_token_map[model]

    num_max_completion_tokens = num_max_token - num_prompt_tokens

    if model == "gpt-4-0125-preview" or model == "gpt-4-turbo" or model == "gpt-4o-mini":
        num_max_completion_tokens = min(num_max_completion_tokens, 4096)
    return num_max_completion_tokens


@retry(wait=wait_exponential(min=10, max=300), stop=stop_after_attempt(10))
async def chat_completion_request(messages, model="gpt-3.5-turbo-16k", model_config_dict: Dict = None):
    """
    Send a chat completion request to the OpenAI API.

    Args:
        messages (list): List of message dictionaries.
        model (str): The name of the model to use.
        model_config_dict (dict): Configuration options for the model.

    Returns:
        dict: The API response.

    Raises:
        Exception: If there's an error in the API call.
    """
    if "claude" in model:
        response = client.chat.completions.create(messages=messages, 
                                                  model=model,
                                                  temperature=0.2).model_dump()
        return response

    num_max_completion_tokens = calc_max_token(messages, model)

    if model_config_dict is None:
        model_config_dict = {
            "temperature": 0.2,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "logit_bias": {},
        }

    json_data = {
        "model": model,
        "messages": messages,
        "max_tokens": num_max_completion_tokens,
        "temperature": model_config_dict["temperature"],
        "top_p": model_config_dict["top_p"],
        "n": model_config_dict["n"],
        "stream": model_config_dict["stream"],
        "frequency_penalty": model_config_dict["frequency_penalty"],
        "presence_penalty": model_config_dict["presence_penalty"],
        "logit_bias": model_config_dict["logit_bias"],
    }

    try:
        response = await asyncio.to_thread(client.chat.completions.create, **json_data)
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response. " + f"OpenAI calling Exception: {e}")
        print(traceback.format_exc())
        raise Exception()


async def chat_completion_request_woretry(messages, model="gpt-3.5-turbo-16k", model_config_dict: Dict = None):
    num_max_completion_tokens = calc_max_token(messages, model)

    if model_config_dict is None:
        model_config_dict = {
            "temperature": 0.2,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "logit_bias": {},
        }

    json_data = {
        "model": model,
        "messages": messages,
        "max_tokens": num_max_completion_tokens,
        "temperature": model_config_dict["temperature"],
        "top_p": model_config_dict["top_p"],
        "n": model_config_dict["n"],
        "stream": model_config_dict["stream"],
        "frequency_penalty": model_config_dict["frequency_penalty"],
        "presence_penalty": model_config_dict["presence_penalty"],
        "logit_bias": model_config_dict["logit_bias"],
    }

    try:
        response = await asyncio.to_thread(client.chat.completions.create, **json_data)
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response. " + f"OpenAI calling Exception: {e}")
        print(traceback.format_exc())
        raise Exception()


async def query_gpt(prompt, woretry=False, temperature=0.2):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant that provides concise titles and summaries.'},
        {'role': 'user', 'content': f"Please provide a concise title for the following text:\n\n{prompt}\n\nYou must return only the title."}
    ]
    model_config_dict = {
        "temperature": temperature,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logit_bias": {},
    }
    if woretry:
        response = await chat_completion_request_woretry(messages, model_config_dict=model_config_dict)
    else:
        response = await chat_completion_request(messages, model_config_dict=model_config_dict)
    response_text = response.choices[0].message.content
    return response_text


async def query_gpt4(prompt, woretry=False, temperature=0.2):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant that provides concise titles and summaries.'},
        {'role': 'user', 'content': f"Please provide a concise title for the following text:\n\n{prompt}\n\nYou must return only the title."}
    ]
    model_config_dict = {
        "temperature": temperature,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logit_bias": {},
    }
    if woretry:
        response = await chat_completion_request_woretry(messages,
                                                   model="gpt-4o-mini",
                                                   model_config_dict=model_config_dict)
    else:
        response = await chat_completion_request(messages,
                                           model="gpt-4o-mini",
                                           model_config_dict=model_config_dict)
    response_text = response.choices[0].message.content
    return response_text


async def query_claude(prompt, woretry=False, temperature=0.2):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant that provides concise titles and summaries.'},
        {'role': 'user', 'content': f"Please provide a concise title for the following text:\n\n{prompt}\n\nYou must return only the title."}
    ]
    model_config_dict = {
        "temperature": temperature,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logit_bias": {},
    }
    if woretry:
        response = await chat_completion_request_woretry(messages,
                                                   model="claude-3-sonnet-20240229",
                                                   model_config_dict=model_config_dict)
    else:
        response = await chat_completion_request(messages,
                                           model="claude-3-sonnet-20240229",
                                           model_config_dict=model_config_dict)
    response_text = response['choices'][0]['message']['content']
    return response_text