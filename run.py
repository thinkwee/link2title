import re
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
import yaml
from asyncio import Semaphore
import sys
import nest_asyncio
from urllib.parse import urlparse

# Import all backend modules
from backend import third_party, gpt, ollama

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

backend_provider = config["backend"]["provider"]

async def query_llm(prompt):
    """
    Query the selected language model based on the backend provider.

    Args:
        prompt (str): The input prompt for the language model.

    Returns:
        str: The response from the language model.

    Raises:
        ValueError: If an unsupported backend provider is specified.
    """
    if backend_provider == "ollama":
        return await ollama.query_ollama(prompt)
    elif backend_provider == "gpt3":
        return await gpt.query_gpt(prompt)
    elif backend_provider == "gpt4":
        return await gpt.query_gpt4(prompt)
    elif backend_provider == "deepseek":
        return await third_party.query_deepseek(prompt)
    elif backend_provider == "qwen":
        return await third_party.query_qwen(prompt)
    elif backend_provider == "ernie":
        return await third_party.query_ernie(prompt)
    elif backend_provider == "glm":
        return await third_party.query_glm(prompt)
    elif backend_provider == "spark":
        return await third_party.query_spark(prompt)
    elif backend_provider == "hunyuan":
        return await third_party.query_hunyuan(prompt)
    else:
        raise ValueError(f"Unsupported backend provider: {backend_provider}")

async def fetch(session, url):
    """
    Fetch content from a given URL using an aiohttp session.

    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        url (str): The URL to fetch.

    Returns:
        str: The content of the URL.
    """
    async with session.get(url, timeout=10) as response:
        return await response.text()

MAX_CONCURRENT_REQUESTS = 3  
semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(10))
async def jina_get_text(session, url):
    """
    Fetch text content from a URL using Jina AI's content extraction service.

    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        url (str): The URL to fetch.

    Returns:
        str: The extracted text content.

    Raises:
        Exception: If there's an error fetching the URL.
    """
    modified_url = f"https://r.jina.ai/{url}"
    try:
        async with semaphore:
            content = await fetch(session, modified_url)
            await asyncio.sleep(2)  
        return content
    except Exception as e:
        print(f"Error fetching {url}\n {str(e)}")
        raise

def is_valid_url(url):
    """
    Check if a given string is a valid URL.

    Args:
        url (str): The string to check.

    Returns:
        bool: True if the string is a valid URL, False otherwise.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

async def get_title_for_url(session, url):
    """
    Get a title for a given URL.

    Args:
        session (aiohttp.ClientSession): The aiohttp session.
        url (str): The URL to get a title for.

    Returns:
        str: The generated title for the URL.
    """
    try:
        doc = await jina_get_text(session, url)
        title = await query_llm(doc[:1000])
        print(f"Processed: [{title}] for [{url}]")
        return title.strip()
    except Exception as e:
        print(f"Error getting title for {url}: {str(e)}")
        return url

async def process_markdown(input_file, output_file):
    """
    Process a markdown file by replacing URLs with titled links.

    Args:
        input_file (str): Path to the input markdown file.
        output_file (str): Path to the output markdown file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    processed_lines = []
    in_code_block = False
    
    async with aiohttp.ClientSession() as session:
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
            
            if not in_code_block:
                urls = re.findall(r'(?<!\]\()https?://\S+(?<!\))', line)
                for url in urls:
                    if is_valid_url(url):
                        title = await get_title_for_url(session, url)
                        title = title.strip('"')
                        line = line.replace(url, f'[{title}]({url})')
            
            processed_lines.append(line)

    processed_content = '\n'.join(processed_lines)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed_content)

    print(f"Processed markdown has been written to {output_file}")

async def main():
    """
    Main function to run the markdown processing script.
    """
    if len(sys.argv) != 2:
        print("Usage: python run.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = 'output.md'
    await process_markdown(input_file, output_file)

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())