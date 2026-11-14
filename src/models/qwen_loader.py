import time
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from ..utils.seed import resolve_device, seed_everything


class QwenVQA:
    def __init__(self, model_id="Qwen/Qwen3-VL-2B-Instruct", revision="main", device="auto",
                 max_new_tokens=64, min_pixels=4096, max_pixels=262144, local_files_only=True, seed=42,
                 answer_style="short"):
        seed_everything(seed)
        self.device = resolve_device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device.type == "cuda" else torch.float32)
        self.model_id, self.revision = model_id, revision
        self.max_new_tokens = max_new_tokens
        if answer_style not in {"short", "raw"}:
            raise ValueError("answer_style must be short or raw")
        self.system_prompt = "Answer the question using only a short word or phrase. Do not include an explanation." if answer_style == "short" else None
        start = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, local_files_only=local_files_only)
        self.processor.tokenizer.padding_side = "left"
        # Current Qwen3-VL processor: size entries are pixel budgets, not side lengths.
        self.processor.image_processor.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision,
                    dtype=self.dtype, attn_implementation="sdpa", local_files_only=local_files_only).to(self.device)
        self.model.eval().requires_grad_(False)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - start
        self.calls = 0
        self.samples = 0

    @torch.inference_mode()
    def answer(self, images, questions):
        if len(images) != len(questions) or not images:
            raise ValueError("Nonempty matching image/question batches required")
        messages = [[{"role": "user", "content": [{"type": "image", "image": image},
                      {"type": "text", "text": question}]}] for image, question in zip(images, questions)]
        if self.system_prompt:
            messages = [[{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}] + m for m in messages]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt", padding=True).to(self.device)
        # Greedy decoding: do not pass temperature or sampling controls.
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = output[:, inputs["input_ids"].shape[1]:]
        answers = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.calls += 1
        self.samples += len(images)
        if not all(isinstance(a, str) and a.strip() for a in answers):
            raise RuntimeError("Qwen returned empty/invalid decoded answers")
        return answers

    def metadata(self):
        return {"model_id": self.model_id, "revision": self.revision, "dtype": str(self.dtype),
                "device": str(self.device), "padding_side": self.processor.tokenizer.padding_side,
                "image_size": self.processor.image_processor.size, "load_seconds": self.load_seconds,
                "max_new_tokens": self.max_new_tokens, "do_sample": False,
                "system_prompt": self.system_prompt,
                "model_calls": self.calls, "samples": self.samples,
                "frozen": all(not p.requires_grad for p in self.model.parameters())}
