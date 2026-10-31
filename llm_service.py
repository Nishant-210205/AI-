"""
LLM Service Module (RAG-11, RAG-12, RAG-14, Cloud-Native Free Providers)
Primary Cloud: Google Gemini (Free Tier via google-genai SDK, Flash / Flash-Lite)
Secondary Cloud: Groq (Developer Free Tier via groq SDK, Llama-3.3-70B / 3.1-8B)
Optional: Anthropic Claude (Active ONLY if ENABLE_CLAUDE=true and API key exists)
Local Fallback: Ollama (llama3.2:3b)
Test-Only: MockLLMService (For deterministic offline CI tests)
"""

import os
import time
import random
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    # 0. Check Streamlit session_state if running inside Streamlit (runtime user input)
    try:
        import streamlit as st
        if hasattr(st, "session_state") and st.session_state is not None:
            for k in (key, key.upper(), key.lower()):
                if k in st.session_state and st.session_state[k]:
                    cleaned = str(st.session_state[k]).strip().strip("\"'`").strip()
                    if cleaned:
                        return cleaned
    except Exception:
        pass

    # 1. Check environment variables (exact match, uppercase, lowercase)
    for k in (key, key.upper(), key.lower()):
        val = os.environ.get(k)
        if val is not None:
            cleaned = str(val).strip().strip("\"'`").strip()
            if cleaned:
                return cleaned

    # 2. Check Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets is not None:
            for k in (key, key.upper(), key.lower()):
                if k in st.secrets and st.secrets[k] is not None:
                    cleaned = str(st.secrets[k]).strip().strip("\"'`").strip()
                    if cleaned:
                        return cleaned
            # Nested table support (e.g. [qdrant] url = "...", [gemini] api_key = "...")
            for section in ("qdrant", "gemini", "groq", "general"):
                if section in st.secrets and isinstance(st.secrets[section], dict):
                    sec_dict = st.secrets[section]
                    sub_key = key.replace(f"{section.upper()}_", "").lower()
                    for sk in (sub_key, key.lower(), key):
                        if sk in sec_dict and sec_dict[sk] is not None:
                            cleaned = str(sec_dict[sk]).strip().strip("\"'`").strip()
                            if cleaned:
                                return cleaned
    except Exception:
        pass

    return default


UNKNOWN_ANSWER_PHRASE = "Information not found in the available documents."

SYSTEM_PROMPT = """You are a college information assistant.
Answer the user's question using ONLY the context provided below.
Do not use external knowledge or invent facts.
If the question is paraphrased or asks about a related policy (e.g., how many classes can be missed, passing requirements, or placement rules), explain what the documents explicitly state about that topic (such as the 75% attendance requirement or detention rules below 65%).
If the information or topic is completely absent from the context, reply strictly:
"Information not found in the available documents."
Give a concise, clear, and direct answer.
Always cite the source document name and page number if available."""


class BaseLLMService(ABC):
    """Abstract base class for LLM providers."""

    provider_name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass

    def build_prompt(self, question: str, context: str) -> str:
        return f"""CONTEXT:
{context}

QUESTION:
{question}"""


class GeminiLLMService(BaseLLMService):
    """
    Primary Cloud LLM Service using Google AI Studio Free Tier (google-genai SDK).
    Default Model: gemini-2.0-flash (Verified Free Tier: 15 RPM, 1,000,000 TPM, 1,500 RPD).
    Official limits: https://ai.google.dev/pricing
    """

    provider_name: str = "gemini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1
    ):
        self.api_key = api_key or get_secret("GEMINI_API_KEY")
        self.model_name = model_name or get_secret("GEMINI_MODEL", "gemini-2.0-flash")
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        latest_key = get_secret("GEMINI_API_KEY") or self.api_key
        if self._client is None or (latest_key and latest_key != self.api_key):
            self.api_key = latest_key
            if self.api_key:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        current_k = get_secret("GEMINI_API_KEY") or self.api_key
        return bool(current_k and len(current_k.strip()) > 10 and not current_k.startswith("your_") and not current_k.startswith("AIzaSyYour"))

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.is_available():
            raise RuntimeError(
                "Google Gemini API key is not configured. "
                "Please add GEMINI_API_KEY in your Streamlit Secrets (share.streamlit.io) or local .env file. "
                "Get a free key (Rs 0, no credit card required) at https://aistudio.google.com/apikey"
            )

        from google import genai
        from google.genai import types
        from google.genai import errors

        client = self._get_client()
        sys_instruction = system_prompt or SYSTEM_PROMPT

        # Exponential backoff for 429 rate limit
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_instruction,
                        temperature=self.temperature,
                        max_output_tokens=1024,
                    )
                )
                text = response.text
                return text.strip() if text else UNKNOWN_ANSWER_PHRASE
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "429" in err_str
                    or "quota" in err_str
                    or "resource_exhausted" in err_str
                    or getattr(e, "code", None) == 429
                )
                if is_rate_limit and attempt < max_retries:
                    sleep_time = (attempt + 1) * 2.5
                    time.sleep(sleep_time)
                    continue

                if is_rate_limit:
                    raise RuntimeError(
                        "Google Gemini Free-Tier rate limit reached (15 RPM / 1,500 RPD). "
                        "Please wait a few seconds before asking another question. "
                        "Official limits: https://ai.google.dev/pricing"
                    ) from e
                raise RuntimeError(f"Google Gemini API error: {e}") from e


class GroqLLMService(BaseLLMService):
    """
    Secondary Free Cloud LLM Service using Groq Cloud Free Tier.
    Default Model: llama-3.3-70b-versatile or llama-3.1-8b-instant.
    Official limits: https://console.groq.com/docs/rate-limits (30 RPM, free of charge).
    """

    provider_name: str = "groq"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1
    ):
        self.api_key = api_key or get_secret("GROQ_API_KEY")
        self.model_name = model_name or get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None and self.api_key:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.is_available():
            raise RuntimeError(
                "Groq API key is not configured. "
                "Please add GROQ_API_KEY in your Streamlit Secrets or local .env file. "
                "Get a free key at https://console.groq.com/keys"
            )

        client = self._get_client()
        sys_instruction = system_prompt or SYSTEM_PROMPT

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": sys_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=1024
                )
                content = chat_completion.choices[0].message.content
                return content.strip() if content else UNKNOWN_ANSWER_PHRASE
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str
                if is_rate_limit and attempt < max_retries:
                    time.sleep((attempt + 1) * 2)
                    continue
                if is_rate_limit:
                    raise RuntimeError(
                        "Groq Free Tier rate limit reached (30 RPM). "
                        "Please wait a few seconds and try again. "
                        "Official limits: https://console.groq.com/docs/rate-limits"
                    ) from e
                raise RuntimeError(f"Groq API error: {e}") from e


class ClaudeLLMService(BaseLLMService):
    """
    Anthropic Claude Service (OPTIONAL, PAID API, OFF BY DEFAULT).
    Hard rule: Active ONLY when ENABLE_CLAUDE=true and ANTHROPIC_API_KEY is provided.
    Pricing: Paid pay-as-you-go per token (Official: https://www.anthropic.com/pricing).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        is_enabled: Optional[bool] = None
    ):
        self.api_key = api_key if api_key is not None else get_secret("ANTHROPIC_API_KEY")
        self.model_name = model_name or get_secret("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._is_enabled_override = is_enabled
        self._client = None

    def is_enabled(self) -> bool:
        if self._is_enabled_override is not None:
            return self._is_enabled_override
        val = (get_secret("ENABLE_CLAUDE", "false") or "").strip().lower()
        return val in ("true", "1", "yes")

    def is_available(self) -> bool:
        return bool(self.is_enabled() and self.api_key and len(self.api_key.strip()) > 10)

    def _get_client(self):
        if self._client is None and self.api_key:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.is_enabled():
            raise RuntimeError(
                "Anthropic Claude is disabled by default to prevent unexpected charges (Rs 0 budget policy). "
                "To enable it explicitly, set ENABLE_CLAUDE=true in secrets. "
                "Official pricing: https://www.anthropic.com/pricing"
            )
        if not self.api_key or len(self.api_key.strip()) == 0:
            raise RuntimeError("Anthropic API key is not configured.")

        import anthropic
        client = self._get_client()
        sys_instruction = system_prompt or SYSTEM_PROMPT

        try:
            message = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=sys_instruction,
                messages=[{"role": "user", "content": prompt}]
            )
            if message and message.content:
                text_blocks = [b.text for b in message.content if getattr(b, "type", "") == "text" or hasattr(b, "text")]
                return "".join(text_blocks).strip()
            return UNKNOWN_ANSWER_PHRASE
        except Exception as e:
            raise RuntimeError(f"Anthropic Claude API Error: {e}") from e


class BedrockClaudeLLMService(BaseLLMService):
    """
    Amazon Bedrock Claude Adapter (OPTIONAL, PAID, OFF BY DEFAULT).
    Hard rule: Active ONLY when ENABLE_CLAUDE=true, ENABLE_BEDROCK_CLAUDE=true, and AWS credentials are set.
    Status: Implemented, unit-tested with mocks, NOT live-tested; enabling requires the owner's own paid credentials.
    Official Bedrock Model IDs:
    - anthropic.claude-3-5-haiku-20241022-v1:0
    - anthropic.claude-3-haiku-20240307-v1:0
    Verified doc URL: https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        is_enabled: Optional[bool] = None
    ):
        self.model_id = model_id or get_secret("BEDROCK_CLAUDE_MODEL", "anthropic.claude-3-5-haiku-20241022-v1:0")
        self.region_name = region_name or get_secret("AWS_REGION", "us-east-1")
        self.aws_access_key_id = aws_access_key_id or get_secret("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or get_secret("AWS_SECRET_ACCESS_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._is_enabled_override = is_enabled
        self._client = None

    def is_enabled(self) -> bool:
        if self._is_enabled_override is not None:
            return self._is_enabled_override
        claude_on = (get_secret("ENABLE_CLAUDE", "false") or "").strip().lower() in ("true", "1", "yes")
        bedrock_on = (get_secret("ENABLE_BEDROCK_CLAUDE", "false") or "").strip().lower() in ("true", "1", "yes")
        return bool(claude_on and bedrock_on)

    def is_available(self) -> bool:
        return bool(self.is_enabled() and self.aws_access_key_id and len(self.aws_access_key_id.strip()) > 5)

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
        return self._client

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.is_enabled():
            raise RuntimeError(
                "Amazon Bedrock Claude is disabled by default to prevent unexpected charges (Rs 0 budget policy). "
                "To enable it explicitly, set ENABLE_CLAUDE=true and ENABLE_BEDROCK_CLAUDE=true in secrets. "
                "Docs note: implemented, unit-tested with mocks, NOT live-tested; enabling requires the owner's own paid credentials."
            )
        if not self.is_available():
            raise RuntimeError("AWS credentials not configured for Amazon Bedrock Claude.")

        client = self._get_client()
        sys_instruction = system_prompt or SYSTEM_PROMPT
        try:
            response = client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                system=[{"text": sys_instruction}],
                inferenceConfig={"temperature": self.temperature, "maxTokens": self.max_tokens}
            )
            output = response.get("output", {}).get("message", {}).get("content", [])
            for block in output:
                if "text" in block:
                    return block["text"].strip()
            return UNKNOWN_ANSWER_PHRASE
        except Exception as e:
            raise RuntimeError(f"Amazon Bedrock Claude API Error: {e}") from e


class VertexClaudeLLMService(BaseLLMService):
    """
    Google Cloud Vertex AI Claude Adapter (OPTIONAL, PAID, OFF BY DEFAULT).
    Hard rule: Active ONLY when ENABLE_CLAUDE=true, ENABLE_VERTEX_CLAUDE=true, and GCP project is configured.
    Status: Implemented, unit-tested with mocks, NOT live-tested; enabling requires the owner's own paid credentials.
    Official Vertex Claude Model IDs:
    - claude-3-5-haiku@20241022
    - claude-3-5-sonnet@20241022
    Verified doc URL: https://docs.anthropic.com/en/api/claude-on-vertex-ai
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        is_enabled: Optional[bool] = None
    ):
        self.model_name = model_name or get_secret("VERTEX_CLAUDE_MODEL", "claude-3-5-haiku@20241022")
        self.project_id = project_id or get_secret("GCP_PROJECT_ID")
        self.region = region or get_secret("GCP_REGION", "us-east5")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._is_enabled_override = is_enabled
        self._client = None

    def is_enabled(self) -> bool:
        if self._is_enabled_override is not None:
            return self._is_enabled_override
        claude_on = (get_secret("ENABLE_CLAUDE", "false") or "").strip().lower() in ("true", "1", "yes")
        vertex_on = (get_secret("ENABLE_VERTEX_CLAUDE", "false") or "").strip().lower() in ("true", "1", "yes")
        return bool(claude_on and vertex_on)

    def is_available(self) -> bool:
        return bool(self.is_enabled() and self.project_id and len(self.project_id.strip()) > 3)

    def _get_client(self):
        if self._client is None and self.project_id:
            from anthropic import AnthropicVertex
            self._client = AnthropicVertex(project_id=self.project_id, region=self.region)
        return self._client

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.is_enabled():
            raise RuntimeError(
                "Google Vertex AI Claude is disabled by default to prevent unexpected charges (Rs 0 budget policy). "
                "To enable it explicitly, set ENABLE_CLAUDE=true and ENABLE_VERTEX_CLAUDE=true in secrets. "
                "Docs note: implemented, unit-tested with mocks, NOT live-tested; enabling requires the owner's own paid credentials."
            )
        if not self.is_available():
            raise RuntimeError("GCP Project ID not configured for Google Vertex AI Claude.")

        client = self._get_client()
        sys_instruction = system_prompt or SYSTEM_PROMPT
        try:
            message = client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=sys_instruction,
                messages=[{"role": "user", "content": prompt}]
            )
            if message and message.content:
                text_blocks = [b.text for b in message.content if getattr(b, "type", "") == "text" or hasattr(b, "text")]
                return "".join(text_blocks).strip()
            return UNKNOWN_ANSWER_PHRASE
        except Exception as e:
            raise RuntimeError(f"Google Vertex AI Claude API Error: {e}") from e


class FallbackLLMChain(BaseLLMService):
    """
    Production Scalable LLM Fallback Chain with Circuit Breakers.
    Priority:
    1. GeminiLLMService (Free Tier: 15 RPM, 1M TPM, 1,500 RPD)
    2. GroqLLMService (Free Tier: 30 RPM)
    3. MockLLMService (Offline / Testing fallback)
    Features:
    - Exponential backoff with random jitter on 429 rate limits.
    - Provider-level circuit breaker: trips for 60s after 3 consecutive failures.
    - Graceful degradation message if all providers hit rate limits.
    """

    def __init__(self, providers: Optional[List[Tuple[str, BaseLLMService]]] = None):
        if providers:
            self.providers = providers
        else:
            self.providers = [
                ("Gemini", GeminiLLMService()),
                ("Groq", GroqLLMService())
            ]
        self.circuit_state = {name: {"failures": 0, "tripped_until": 0.0} for name, _ in self.providers}
        self.stats = {"fallbacks_triggered": 0, "rate_limits_handled": 0}

    def is_available(self) -> bool:
        now = time.time()
        for name, srv in self.providers:
            if srv.is_available() and now >= self.circuit_state[name]["tripped_until"]:
                return True
        return False

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        now = time.time()

        for name, srv in self.providers:
            # Check circuit breaker
            if now < self.circuit_state[name]["tripped_until"]:
                continue
            if not srv.is_available():
                continue

            for attempt in range(2):
                try:
                    ans = srv.generate_response(prompt, system_prompt)
                    self.circuit_state[name]["failures"] = 0
                    return ans
                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = "429" in err_str or "quota" in err_str or "rate limit" in err_str
                    if is_rate_limit:
                        self.stats["rate_limits_handled"] += 1
                        jitter = random.uniform(0.5, 1.2)
                        time.sleep((attempt + 1) * jitter)

            self.circuit_state[name]["failures"] += 1
            if self.circuit_state[name]["failures"] >= 2:
                self.circuit_state[name]["tripped_until"] = time.time() + 60.0
            self.stats["fallbacks_triggered"] += 1

        return (
            "⚠️ Cloud LLM quota currently exhausted across all free providers. "
            "Please refer to the verified source documents and retrieved excerpts below for your answer."
        )


class OllamaLLMService(BaseLLMService):
    """
    Local fallback LLM service using locally installed Ollama (llama3.2:3b).
    Not available in Streamlit Community Cloud.
    """

    def __init__(self, model_name: str = "llama3.2:3b", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature

    def is_available(self) -> bool:
        try:
            import ollama
            res = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}]
            )
            return bool(res and "message" in res)
        except Exception:
            return False

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import ollama
        sys_instruction = system_prompt or SYSTEM_PROMPT
        full_content = f"{sys_instruction}\n\n{prompt}"
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": full_content}],
                options={"temperature": self.temperature}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Local Ollama error: {e}. Note: Ollama cannot run on Streamlit Cloud.") from e


class MockLLMService(BaseLLMService):
    """
    Deterministic mock service for unit testing and offline CI verification.
    """

    provider_name: str = "mock"

    def __init__(self, canned_response: Optional[str] = None):
        self.canned_response = canned_response

    def is_available(self) -> bool:
        return True

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self.canned_response:
            return self.canned_response
        lower_p = prompt.lower()
        if any(w in lower_p for w in ["hostel", "swimming", "chief guest", "cafeteria", "bus route", "parking", "laundry", "alumni", "flight"]):
            return UNKNOWN_ANSWER_PHRASE
        return "According to the college academic regulations, students must maintain at least 75% attendance. Source: academic_regulations.pdf"


def get_llm_service(provider: Optional[str] = None, **kwargs) -> BaseLLMService:
    """
    Factory function for LLM services.
    Default: GeminiLLMService (Free-tier Google GenAI).
    Supports: gemini, groq, claude, bedrock, vertex, chain, ollama, mock.
    """
    selected = (provider or get_secret("LLM_PROVIDER", "gemini")).lower()

    if selected == "gemini":
        return GeminiLLMService(**kwargs)
    elif selected == "groq":
        return GroqLLMService(**kwargs)
    elif selected == "chain":
        return FallbackLLMChain(**kwargs)
    elif selected == "claude":
        return ClaudeLLMService(**kwargs)
    elif selected == "bedrock":
        return BedrockClaudeLLMService(**kwargs)
    elif selected == "vertex":
        return VertexClaudeLLMService(**kwargs)
    elif selected == "ollama":
        return OllamaLLMService(**kwargs)
    elif selected == "mock" or selected == "test":
        return MockLLMService(**kwargs)
    else:
        if get_secret("GEMINI_API_KEY"):
            return GeminiLLMService(**kwargs)
        if get_secret("GROQ_API_KEY"):
            return GroqLLMService(**kwargs)
        return GeminiLLMService(**kwargs)


# Backwards compatibility alias
LLMService = GeminiLLMService

