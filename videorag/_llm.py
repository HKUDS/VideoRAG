import numpy as np

from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError
from ollama import AsyncClient

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage

global_openai_async_client = None
global_azure_openai_async_client = None
global_ollama_client = None

def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client

def get_ollama_async_client_instance():
    global global_ollama_client
    if global_ollama_client is None:
        # set OLLAMA_HOST
        global_ollama_client = AsyncClient(host="http://10.0.1.12:11434")  # Adjust base URL if necessary        
    return global_ollama_client

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)

# Setup LLM Configuration.
@dataclass
class LLMConfig:
    embedding_func: EmbeddingFunc
    embedding_batch_num: int 
    embedding_func_max_async: int 
    query_better_than_threshold: float
    
    best_model_func: callable 
    best_model_max_token_size: int
    best_model_max_async: int
    
    cheap_model_func: callable
    cheap_model_max_token_size: int
    cheap_model_max_async: int

##### OpenAI Configuration

    
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


openai_config = LLMConfig(
    embedding_func = field(default_factory=lambda: openai_embedding)
    embedding_batch_num = 32
    embedding_func_max_async = 16
    query_better_than_threshold = 0.2

    # LLM        
    best_model_func = gpt_4o_mini_complete
    best_model_max_token_size = 32768
    best_model_max_async = 16
        
    cheap_model_func = gpt_4o_mini_complete
    cheap_model_max_token_size = 32768
    cheap_model_max_async = 16

###### Azure OpenAI Configuration
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )



@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


azure_openai_config = LLMConfig(
    embedding_func = field(default_factory=lambda: azure_openai_embedding),
    embedding_batch_num = 32
    embedding_func_max_async = 16
    query_better_than_threshold = 0.2

    best_model_func: callable = azure_gpt_4o_complete
    best_model_max_token_size = 32768
    best_model_max_async = 16

    cheap_model_func: callable = azure_gpt_4o_mini_complete
    cheap_model_max_token_size = 32768
    cheap_model_max_async = 16


######  Ollama configuration

async def ollama_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Initialize the Ollama client
    ollama_client = get_ollama_async_client_instance()

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # Send the request to Ollama
    response = await ollama_client.chat(
        model=model,
        messages=messages
    )
    # print(messages)
    # print(response['message']['content'])

    
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response['message']['content'], "model": model}}
        )
        await hashing_kv.index_done_callback()

    return response['message']['content']


async def ollama_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await ollama_complete_if_cache(
        #"deepseek-r1:32b",  # For now select your model
        "gemma2:latest",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages
    )

async def ollama_mini_complete(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await ollama_complete_if_cache(
        # "deepseek-r1:latest",  # For now select your model
        "olmo2",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages
    )

@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def ollama_embedding(texts: list[str]) -> np.ndarray:
    # Initialize the Ollama client
    ollama_client = get_ollama_async_client_instance()

    # Send the request to Ollama for embeddings
    response = await ollama_client.embed(
        model="nomic-embed-text",  # Replace with the appropriate Ollama embedding model
        input=texts
    )

    # Extract embeddings from the response
    embeddings = response['embeddings']

    return np.array(embeddings)

ollama_config = LLMConfig(
    embedding_func= EmbeddingFunc = field(default_factory=lambda: ollama_embedding),
    embedding_batch_num = 1,
    embedding_func_max_async = 1,
    query_better_than_threshold = 0.2,
    best_model_func = ollama_complete ,   
    best_model_max_token_size: int = 32768,
    best_model_max_async  = 1,
    cheap_model_func = ollama_mini_complete,
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 1)
