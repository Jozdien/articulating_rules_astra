import os
from anthropic import Anthropic, AsyncAnthropic
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from .model_registry import get_provider

load_dotenv()


class ModelClient:
    def __init__(self):
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.openrouter = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.async_anthropic = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.async_openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.async_openrouter = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

    def query(self, model_name: str, messages: list[dict], temperature: float = 0.7) -> str:
        provider, model_id = get_provider(model_name)

        if provider == "anthropic":
            return self._query_anthropic(model_id, messages, temperature)
        elif provider in ["openai", "openrouter"]:
            client = self.openai if provider == "openai" else self.openrouter
            return self._query_openai(client, model_id, messages, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _query_anthropic(self, model: str, messages: list[dict], temperature: float) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]

        kwargs = {"model": model, "messages": user_messages, "max_tokens": 4096, "temperature": temperature}
        if system:
            kwargs["system"] = system

        response = self.anthropic.messages.create(**kwargs)
        return response.content[0].text

    def _query_openai(self, client: OpenAI, model: str, messages: list[dict], temperature: float) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    async def query_async(self, model_name: str, messages: list[dict], temperature: float = 0.7) -> str:
        provider, model_id = get_provider(model_name)

        if provider == "anthropic":
            return await self._query_anthropic_async(model_id, messages, temperature)
        elif provider in ["openai", "openrouter"]:
            client = self.async_openai if provider == "openai" else self.async_openrouter
            return await self._query_openai_async(client, model_id, messages, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _query_anthropic_async(self, model: str, messages: list[dict], temperature: float) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]

        kwargs = {"model": model, "messages": user_messages, "max_tokens": 4096, "temperature": temperature}
        if system:
            kwargs["system"] = system

        response = await self.async_anthropic.messages.create(**kwargs)
        return response.content[0].text

    async def _query_openai_async(self, client: AsyncOpenAI, model: str, messages: list[dict], temperature: float) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
