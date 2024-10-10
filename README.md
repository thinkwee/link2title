# Link2Title

This project is a Python script that processes Markdown files, automatically replacing plain URLs with titled links. It fetches the content of each URL, generates a descriptive title using a Language Model (LLM), and updates the Markdown file with the new, more informative links.

## Features

- Asynchronous processing of URLs for improved performance
- Supports multiple backend LLM providers
- Uses Jina AI's content extraction service for reliable web scraping
- Implements retry logic and rate limiting to handle network issues

## Requirements

- Python 3.7+
- Required Python packages (see `requirements.txt`)

## Configuration

Edit the `config.yaml` file to set your preferred LLM backend provider and any necessary API keys.

## Usage

1. Configure your preferred LLM backend and API key in the `config.yaml` file.

2. Run the script with the input Markdown file as an argument:
```
python3 run.py input_file.md
```

The processed file will be saved as `output.md` in the same directory.

## Example

Just try it on this README.md, say running
```
python3 run.py README.md
```
See what happens.to following links

-  https://www.markdownguide.org/
-  https://docs.python.org/3/library/asyncio.html
-  https://en.wikipedia.org/wiki/Language_model

After running `run.py` on this README, these links would be transformed into more informative, titled links. The output might look like this:

- [Markdown Guide - Basic Syntax, Extended Syntax, Cheat Sheet](https://www.markdownguide.org/)
- [asyncio — Asynchronous I/O — Python 3.11.5 documentation](https://docs.python.org/3/library/asyncio.html)
- [Language model - Wikipedia](https://en.wikipedia.org/wiki/Language_model)

## Supported LLM Backends

- OpenAI (GPT3/4)
- Ollama
- DeepSeek
- Qwen
- ERNIE
- GLM
- Spark
- HunYuan

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.