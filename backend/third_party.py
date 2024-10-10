from openai import OpenAI
import os
import yaml
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
import qianfan
import asyncio

# Load configuration
file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)
global_config = yaml.safe_load(open(os.path.join(project_path, "config.yaml"), "r"))

# Load API keys from configuration
DEEPSEEK_API_KEY = global_config.get("backend").get("deepseek_api_key")
QWEN_API_KEY = global_config.get("backend").get("qwen_api_key")
QIANFAN_ACCESS_KEY = global_config.get("backend").get("QIANFAN_ACCESS_KEY")
QIANFAN_SECRET_KEY = global_config.get("backend").get("QIANFAN_SECRET_KEY")
GLM_API_KEY = global_config.get("backend").get("glm_api_key")
HUNYUAN_API_KEY = global_config.get("backend").get("hunyuan_api_key")
SPARK_API_KEY = global_config.get("backend").get("spark_api_key")

# Set environment variables
os.environ["DASHSCOPE_API_KEY"] = QWEN_API_KEY
os.environ["QIANFAN_ACCESS_KEY"] = QIANFAN_ACCESS_KEY
os.environ["QIANFAN_SECRET_KEY"] = QIANFAN_SECRET_KEY

# Initialize clients for different models
client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
client_deepseek_llama_index = OpenAILike(api_key=DEEPSEEK_API_KEY, api_base="https://api.deepseek.com/beta", model="deepseek-chat")

client_qwen = OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
client_qwen_llama_index = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX, api_key=QWEN_API_KEY)

client_glm = OpenAI(api_key=GLM_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/") 
client_glm_llama_index = OpenAILike(api_key=GLM_API_KEY, api_base="https://open.bigmodel.cn/api/paas/v4/", model="glm-4-flash", is_chat_model=True, is_function_calling_model=False)

client_hunyuan = OpenAI(api_key=HUNYUAN_API_KEY, base_url="https://api.hunyuan.cloud.tencent.com/v1")
client_hunyuan_llama_index = OpenAILike(api_key=HUNYUAN_API_KEY, api_base="https://api.hunyuan.cloud.tencent.com/v1", model="hunyuan-lite", is_chat_model=True, is_function_calling_model=False)

client_spark = OpenAI(api_key=SPARK_API_KEY, base_url='https://spark-api-open.xf-yun.com/v1')
client_spark_llama_index = OpenAILike(api_key=SPARK_API_KEY, api_base='https://spark-api-open.xf-yun.com/v1', model="general", is_chat_model=True, is_function_calling_model=False)

async def query_deepseek(prompt):
    """
    Query the DeepSeek model with a given prompt.

    Args:
        prompt (str): The input text to generate a title for.

    Returns:
        str: The generated title.
    """
    response = await asyncio.to_thread(client_deepseek.chat.completions.create,
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides concise titles and summaries."},
            {"role": "user", "content": f"Please provide a concise title for the following text:\n\n{prompt}\n\nYou must return only the title."}
        ],
        stream=False
    )

    return response.choices[0].message.content

async def query_qwen(prompt, model="qwen-max-latest"):
    """
    Query the Qwen model with a given prompt.

    Args:
        prompt (str): The input text to generate a title for.
        model (str): The specific Qwen model to use. Defaults to "qwen-max-latest".

    Returns:
        str: The generated title.
    """
    completion = await asyncio.to_thread(client_qwen.chat.completions.create,
        model=model,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant that provides concise titles and summaries.'},
            {'role': 'user', 'content': f"Please provide a concise title for the following text:\n\n{prompt}\n\nYou must return only the title."}
        ],
    )
        
    return completion.choices[0].message.content

async def query_ernie(prompt):
    """
    Query the ERNIE model with a given prompt.

    Args:
        prompt (str): The input text to generate a title for.

    Returns:
        str: The generated title.
    """
    chat_comp = qianfan.ChatCompletion()

    resp = await asyncio.to_thread(chat_comp.do, model="ERNIE-Speed-128K", messages=[
        {"role": "user", "content": "You are a helpful assistant that provides concise titles and summaries." + f"Please provide a concise title for the following text:\n\n{prompt}\n\nYou must return only the title."},
    ])

    return resp["body"]["result"]

async def query_glm(prompt):
    """
    Query the GLM model with a given prompt.

    Args:
        prompt (str): The input text to generate a title for.

    Returns:
        str: The generated title.
    """
    completion = await asyncio.to_thread(client_glm.chat.completions.create,
        model="glm-4-flash",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides concise titles and summaries."},
            {"role": "user", "content": f"Please provide a concise title for the following text:\n\n{prompt}\n\nYou must return only the title."}],
            top_p=0.7,
            temperature=0.9
    ) 
 
    return completion.choices[0].message.content

async def query_hunyuan(prompt):
    """
    Query the Hunyuan model with a given prompt.

    Args:
        prompt (str): The input text to generate a title for.

    Returns:
        str: The generated title.
    """
    completion = await asyncio.to_thread(client_hunyuan.chat.completions.create,
        model="hunyuan-lite",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides concise titles and summaries."},
            {"role": "user", "content": f"Please provide a concise title for the following text:\n\n{prompt}\n\nYou must return only the title."}
        ]
    )
    return completion.choices[0].message.content

async def query_spark(prompt):
    """
    Query the Spark model with a given prompt.

    Args:
        prompt (str): The input text to generate a title for.

    Returns:
        str: The generated title.
    """
    completion = await asyncio.to_thread(client_spark.chat.completions.create,
        model='general',
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides concise titles and summaries."},
            {"role": "user", "content": f"Please provide a concise title for the following text:\n\n{prompt}\n\nYou must return only the title."}
        ]
    )
    return completion.choices[0].message.content