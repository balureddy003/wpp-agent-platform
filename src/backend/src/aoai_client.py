# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import json
from typing import Callable
from openai import AzureOpenAI
from azure.core.credentials import TokenCredential
from azure.identity import get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from utils import get_azure_credential
from pathlib import Path

DEFAULT_RAG_GROUNDING_PROMPT = (
    "Use the SOURCES below to answer the USER QUERY.\n"
    "Only use information from SOURCES. If the answer is not present, reply \"I don't know\".\n\n"
    "USER QUERY:\n{query}\n\nSOURCES:\n{sources}\n"
)

def get_prompt(
    prompt: str,
    path: str = "prompts/"
) -> str:
    """
    Load prompt.
    Tries several common locations and falls back to a built-in default
    if the file is not found.
    """
    logger = logging.getLogger(__name__)

    candidates = []
    # Caller-provided path first
    if path:
        candidates.append(Path(path) / prompt)

    # File-relative prompts dir
    here = Path(__file__).resolve().parent
    candidates.append(here / "prompts" / prompt)

    # Common project layouts
    candidates.append(Path.cwd() / "src" / "prompts" / prompt)
    candidates.append(Path.cwd() / "prompts" / prompt)

    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as fp:
                return fp.read()
        except FileNotFoundError:
            continue

    logger.warning(f"Prompt file not found: {prompt}; using built-in default.")
    return DEFAULT_RAG_GROUNDING_PROMPT


RAG_GROUNDING_PROMPT = get_prompt("rag_grounding.txt")


class AOAIClient(AzureOpenAI):
    """
    Chat-only AOAI Client.

    AzureOpenAI wrapper with function-calling and RAG support.
    """

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        api_version: str = "2023-12-01-preview",
        scope: str = "https://cognitiveservices.azure.com/.default",
        azure_credential: TokenCredential = None,
        system_message: str = None,
        function_calling: bool = False,
        tools: list = None,
        functions: dict[str, Callable] = None,
        return_functions: bool = False,
        use_rag: bool = False,
        search_client: SearchClient = None
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if not azure_credential:
            azure_credential = get_azure_credential()
        token_provider = get_bearer_token_provider(azure_credential, scope)
        AzureOpenAI.__init__(
            self,
            api_version=api_version,
            azure_ad_token_provider=token_provider,
            azure_endpoint=endpoint
        )

        # Function-calling:
        self.function_calling = function_calling
        self.tools = tools
        self.functions = functions
        self.return_functions = return_functions

        # RAG:
        self.use_rag = use_rag
        self.search_client = search_client

        # General:
        self.deployment = self.model_name = deployment
        self.api_version = api_version
        self.chat_api = True
        self.messages = []

        if system_message:
            # Prepend system message:
            self.messages = [{"role": "system", "content": system_message}]

    def call_functions(
        self,
        language: str,
        id: str
    ) -> list:
        """
        AOAI function calling.

        Returns function-call responses.
        """
        # Call chat API with function-calling enabled:
        response = self.chat.completions.create(
            model=self.deployment,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
        )

        # Process model's response:
        response_message = response.choices[0].message
        self.messages.append(response_message)
        self.logger.info(f"Model response: {response_message}")

        # Handle function calls:
        function_responses = []
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                self.logger.info(f"Function call: {function_name}")
                self.logger.info(f"Function arguments: {function_args}")

                if function_name in self.functions:
                    # All functions require single extracted parameter:
                    func_input = next(iter(function_args.values()))
                    func = self.functions[function_name]
                    func_response = func(func_input, language, id)
                else:
                    func_response = json.dumps({"error": "Unknown function"})

                function_responses.append(func_response)
                self.logger.info(f"Function response: {str(func_response)}")
                self.messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(func_response)
                })
        else:
            self.logger.info("No tool calls made by model.")

        return function_responses

    def generate_rag_prompt(
        self,
        query: str
    ) -> str:
        """
        Generates RAG grounding prompt given query and search client.
        """
        if not self.search_client:
            self.logger.warning("RAG enabled but no SearchClient; falling back to empty sources.")
            return RAG_GROUNDING_PROMPT.format(query=query, sources="")

        self.logger.info("Calling search client")
        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=50,
            fields="text_vector"
        )
        search_results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["title", "chunk"],
            top=5
        )

        sources_formatted = "=================\n".join(
            [f'TITLE: {doc["title"]}, CONTENT: {doc["chunk"]}' for doc in search_results]
        )

        prompt = RAG_GROUNDING_PROMPT.format(
            query=query,
            sources=sources_formatted
        )

        return prompt

    def chat_completion(
        self,
        message: str,
        language: str = None,
        id: str = None
    ) -> str:
        """
        AOAI chat completion.
        """
        # Add user message:
        prompt = self.generate_rag_prompt(message) if self.use_rag else message
        self.messages.append({"role": "user", "content": prompt})

        if self.function_calling:
            function_results = self.call_functions(language=language, id=id)
            if self.return_functions:
                # Return function-call results directly:
                return function_results

        # Call chat API:
        response = self.chat.completions.create(
            model=self.deployment,
            messages=self.messages
        )
        response_message = response.choices[0].message
        self.logger.info(f"Model response: {response_message}")
        self.messages.append(response_message)

        return response_message.content
