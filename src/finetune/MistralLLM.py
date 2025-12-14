from collections.abc import Mapping
from typing import Any

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast


class MistralLLM(LLM):
    """Wraps Mistral LLM as langchain LLM"""
    # Adapted from: https://colab.research.google.com/drive/1e5gJaUtGVvzJP_Nr-sBEL4J2JubC5YrE?usp=sharing#scrollTo=iiefB_6WJqYI
    # See also: https://medium.com/@jorgepardoserrano/building-a-langchain-agent-with-a-self-hosted-mistral-7b-a-step-by-step-guide-85eda2fbf6c2
    model: Any
    tokenizer: LlamaTokenizerFast

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> str:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            tokenize=True,
            # padding=True,
            # truncate=True,
            return_dict=True,
        )
        model_inputs = model_inputs.to(self.model.device)

        # temperature = kwargs.get("temperature", 0)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            # top_k=4,
            # temperature=temperature,
        )
        decoded = self.tokenizer.batch_decode(generated_ids)

        return decoded[0]

    def batch(self, inputs):
        messages = []
        # Collect a list of list of messages.
        for input in inputs:
            messages.append(
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": input},
                ]
            )

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            tokenize=True,
            padding=True,
            truncate=True,
            return_dict=True,
        )
        model_inputs = model_inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            # temperature=1,
        )
        decoded = self.tokenizer.batch_decode(generated_ids)

        return decoded

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}
