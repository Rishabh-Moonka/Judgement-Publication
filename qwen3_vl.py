import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image


class Qwen3VL:

    def __init__(self, device="cuda:0", model_path=None):
        assert torch.cuda.is_available(), "CUDA is not available"

        self.device = torch.device(device)

        if model_path is None:
            model_path = "Qwen/Qwen3-VL-8B-Instruct"

        # -------- PROCESSOR --------
        self.processor = AutoProcessor.from_pretrained(model_path)

        # -------- MODEL --------
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        self.model.to(self.device)
        self.model.eval()

    def generate(self, image_path, prompt):

        # -------- IMAGE --------
        image = Image.open(image_path).convert("RGB")

        # -------- PROMPT --------
        system_prompt = prompt.get("system prompt", "")
        user_prompt = prompt.get("user prompt", "")

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        # -------- APPLY TEMPLATE --------
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # -------- PREPARE INPUTS --------
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        # -------- GENERATION --------
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # -------- DECODE ONLY NEW TOKENS --------
        generated_ids = output_ids[0][input_len:]

        answer = self.processor.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        return answer