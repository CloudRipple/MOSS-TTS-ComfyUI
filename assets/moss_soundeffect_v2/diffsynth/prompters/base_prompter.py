# MOSS-TTS-V15-ComfyUI patch: removed tokenize_long_prompt (unused by the
# Wan audio prompter path); BasePrompter itself is upstream-verbatim.
import torch
from typing import Any



class BasePrompter:
    def __init__(self):
        self.refiners = []
        self.extenders = []


    def load_prompt_refiners(self, model_manager: Any, refiner_classes=[]):
        for refiner_class in refiner_classes:
            refiner = refiner_class.from_model_manager(model_manager)
            self.refiners.append(refiner)

    def load_prompt_extenders(self, model_manager: Any, extender_classes=[]):
        for extender_class in extender_classes:
            extender = extender_class.from_model_manager(model_manager)
            self.extenders.append(extender)


    @torch.no_grad()
    def process_prompt(self, prompt, positive=True):
        if isinstance(prompt, list):
            prompt = [self.process_prompt(prompt_, positive=positive) for prompt_ in prompt]
        else:
            for refiner in self.refiners:
                prompt = refiner(prompt, positive=positive)
        return prompt

    @torch.no_grad()
    def extend_prompt(self, prompt:str, positive=True):
        extended_prompt = dict(prompt=prompt)
        for extender in self.extenders:
            extended_prompt = extender(extended_prompt)
        return extended_prompt
