# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import TextIteratorStreamer
import torch
from threading import Thread
import subprocess
from cog import BasePredictor, Input, ConcatenateIterator


MODEL_URL = "https://weights.replicate.delivery/default/gemma-7b-it.tar"
MODEL_CACHE = "gemma-cache"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CACHE, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )

    def predict(
        self,
        prompt: str = Input(
            description="Prompt to send to the model.",
            default="Write me a poem about Machine Learning.",
        ),
        # system_prompt: str = Input(
        #     description="System prompt to send to the model. This is prepended to the prompt and helps guide system behavior. Should not be blank.",
        #     default=DEFAULT_SYSTEM_PROMPT,
        # ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=256,
        ),
        min_new_tokens: int = Input(
            description="Minimum number of tokens to generate. To disable, set to -1. A word is generally 2-3 tokens.",
            ge=-1,
            default=-1,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.7,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.95,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens",
            default=50,
        ),
        repetition_penalty: float = Input(
            description="A parameter that controls how repetitive text can be. Lower means more repetitive, while higher means less repetitive. Set to 1.0 to disable.",
            ge=0.0,
            default=1,
        ),
        # stop_sequences: str = Input(
        #     description="A comma-separated list of sequences to stop generation at. For example, '<end>,<stop>' will stop generation at the first instance of 'end' or '<stop>'.",
        #     default=None,
        # ),
        # seed: int = Input(
        #     description="Random seed. Leave blank to randomize the seed",
        #     default=None,
        # ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""

        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        with torch.inference_mode():
            thread = Thread(
                target=self.model.generate,
                kwargs=dict(
                    **input_ids.to(self.model.device),
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                ),
            )
            thread.start()
            for new_token in streamer:
                yield new_token
            thread.join()
