"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Litellm model implementation for the MLGym framework.

This module provides a model implementation that uses the Litellm library
to interact with different LLM providers. It handles cost tracking,
context window management, and API interactions.

Adapted from SWE-agent/sweagent/agent/models.py
"""
import os

import litellm
import litellm.types.utils
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from mlgym.backend import _MAX_RETRIES
from mlgym.backend.base import BaseModel, ModelArguments
from mlgym.exceptions import ContextWindowExceededError, CostLimitExceededError
from mlgym.types import HistoryItem

# litellm._turn_on_debug()

class LiteLLMModel(BaseModel):
    def __init__(self, args: ModelArguments):
        """Model served by the `litellm` library."""
        super().__init__(args)
        
        self._setup_client()
        
    
    def _setup_client(self):
        # Extract the model name without the "litellm:" prefix
        self.model_name = self.args.model_name.split(":")[1]
        self.model_max_input_tokens = litellm.model_cost.get(self.model_name, {}).get("max_input_tokens")
        self.model_max_output_tokens = litellm.model_cost.get(self.model_name, {}).get("max_output_tokens")
        self.lm_provider = litellm.model_cost.get(self.model_name, {}).get("litellm_provider")
        
        # Handle Azure configuration - check after removing "litellm:" prefix
        if self.model_name.startswith("azure/"):
            if hasattr(self.args, "api_key") and hasattr(self.args, "host_url"):
                self.logger.info(f"Configuring Azure OpenAI: {self.args.host_url}")
                # Configure Azure-specific settings
                litellm.api_key = self.args.api_key
                litellm.api_base = self.args.host_url
                if hasattr(self.args, "api_version"):
                    litellm.api_version = self.args.api_version
            else:
                self.logger.warning("Azure model specified but missing azure_api_key or azure_endpoint")
        
        # Handle Vertex AI configuration - check after removing "litellm:" prefix
        elif self.model_name.startswith("vertex_ai/"):
            self.logger.info("Configuring Vertex AI")
            if hasattr(self.args, "vertex_project") and hasattr(self.args, "vertex_location", "us-east5"):
                os.environ["GOOGLE_CLOUD_PROJECT"] = self.args.vertex_project
                os.environ["VERTEXAI_PROJECT"] = self.args.vertex_project
                os.environ["VERTEXAI_LOCATION"] = self.args.vertex_location
                
                # Set credentials file if provided
                if hasattr(self.args, "vertex_credentials_path"):
                    self.logger.info(f"Using Vertex AI credentials from: {self.args.vertex_credentials_path}")
            else:
                self.logger.warning("Vertex AI model specified but missing vertex_project or vertex_location")
                
        # General host_url handling for custom endpoints
        elif self.lm_provider is None and self.args.host_url is not None:
            # For custom API endpoints
            self.logger.warning(f"Using a custom API base: {self.args.host_url}. Cost management and context length error checking will not work.")

    def update_stats(self, input_tokens: int, output_tokens: int, cost: float = 0.0) -> float:
        self.stats.total_cost += cost
        self.stats.task_cost += cost
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.api_calls += 1

        # Log updated cost values to std. err
        self.logger.debug(
            f"input_tokens={input_tokens:,}, "
            f"output_tokens={output_tokens:,}, "
            f"instance_cost={self.stats.task_cost:.2f}, "
            f"cost={cost:.2f}",
        )
        self.logger.debug(
            f"total_tokens_sent={self.stats.tokens_sent:,}, "
            f"total_tokens_received={self.stats.tokens_received:,}, "
            f"total_cost={self.stats.total_cost:.2f}, "
            f"total_api_calls={self.stats.api_calls:,}",
        )

        # Check whether total cost or instance cost limits have been exceeded
        if 0 < self.args.total_cost_limit <= self.stats.total_cost:
            self.logger.warning(f"Cost {self.stats.total_cost:.2f} exceeds limit {self.args.total_cost_limit:.2f}")
            msg = "Total cost limit exceeded"
            raise CostLimitExceededError(msg)

        if 0 < self.args.per_instance_cost_limit <= self.stats.task_cost:
            self.logger.warning(
                f"Cost {self.stats.task_cost:.2f} exceeds limit {self.args.per_instance_cost_limit:.2f}"
            )
            msg = "Instance cost limit exceeded"
            raise CostLimitExceededError(msg)

        return cost

    @retry(
        wait=wait_random_exponential(min=60, max=180),
        reraise=True,
        stop=stop_after_attempt(_MAX_RETRIES),
        retry=retry_if_not_exception_type((CostLimitExceededError, RuntimeError, litellm.exceptions.UnsupportedParamsError, litellm.exceptions.NotFoundError, litellm.exceptions.PermissionDeniedError, litellm.exceptions.ContextWindowExceededError, litellm.exceptions.APIError)),
    )
    def query(self, history: list[HistoryItem], is_demonstration: bool = False) -> str:
        messages = self.history_to_messages(history, is_demonstration)
        # ensure that we are not just passing the demostratin (str)
        assert isinstance(messages, list)
        
        input_tokens: int = litellm.utils.token_counter(messages=messages, model=self.model_name)
        if self.model_max_input_tokens is None:
            self.logger.warning(f"No max input tokens found for model {self.model_name!r}")
        elif input_tokens > self.model_max_input_tokens:
            msg = f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}"
            raise ContextWindowExceededError(msg)
        extra_args = {}
        if self.args.host_url:
            extra_args["api_base"] = self.args.host_url

        # Azure-specific configuration
        if self.model_name.startswith("azure/"):
            if hasattr(self.args, "api_version"):
                extra_args["api_version"] = self.args.api_version
            if hasattr(self.args, "api_key"):
                extra_args["api_key"] = self.args.api_key
                
        # Vertex AI-specific configuration - updated prefix to vertex_ai/
        elif self.model_name.startswith("vertex_ai/"):
            if hasattr(self.args, "vertex_project"):
                extra_args["vertex_project"] = self.args.vertex_project
            if hasattr(self.args, "vertex_location"):
                extra_args["vertex_location"] = self.args.vertex_location

        completion_kwargs = self.args.completion_kwargs
        if self.lm_provider == "anthropic":
            completion_kwargs["max_tokens"] = self.model_max_output_tokens
        try:
            response: litellm.types.utils.ModelResponse = litellm.completion(  # type: ignore
                model=self.model_name,
                messages=messages,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                api_version=self.args.api_version,
                **completion_kwargs,
                **extra_args,
            )
        except Exception as e:
            self.logger.exception(f"Error during LLM query: {e}")
            raise e
        choices: litellm.types.utils.Choices = response.choices # type: ignore
        output = choices[0].message.content or ""
        # output_dict = {"message": output}

        # update stats
        cost = litellm.cost_calculator.completion_cost(response)
        output_tokens = litellm.utils.token_counter(text=output, model=self.model_name)
        self.update_stats(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost)

        return output

    def history_to_messages(
        self,
        history: list[HistoryItem],
        is_demonstration: bool = False,
    ) -> str | list[dict[str, str]]:
        if is_demonstration:
            history = [entry for entry in history if entry["role"] != "system"]
            return "\n".join([entry.get("content") or "" for entry in history])
        
        messages = []
        for entry in history:
            messages.append({"role": entry["role"], "content": entry.get("content", "")})
        return messages
