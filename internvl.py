import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig
)
from PIL import Image


class InternVL:
    def __init__(self, device="cuda:0", model_path=None, use_quantization=False):

        self.device = torch.device(device)

        self.model_id = "OpenGVLab/InternVL3_5-8B-HF"

        self.model_path = model_path if model_path else self.model_id

        # -------- PROCESSOR --------
        self.processor = AutoProcessor.from_pretrained(
            self.model_path
        )

        # -------- MODEL --------
        if use_quantization:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                quantization_config=quant_config,
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
            ).to(self.device)

        self.model.eval()

    def generate(self, image_path, prompt, max_new_tokens=256):

        # -------- PROMPT HANDLING --------
        if isinstance(prompt, dict):
            system_prompt = prompt.get("system prompt", "")
            user_prompt = prompt.get("user prompt", "")
        else:
            system_prompt = ""
            user_prompt = prompt

        # -------- IMAGE --------
        image = Image.open(image_path).convert("RGB")

        # -------- CONVERSATION --------
        conversation = []

        if system_prompt:
            conversation.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
            })

        conversation.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}
            ]
        })

        # -------- TEMPLATE --------
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # -------- INPUTS --------
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        # -------- GENERATION --------
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )

        # -------- DECODE ONLY NEW TOKENS --------
        output_tokens = generated_ids[0][input_len:]

        answer_text = self.processor.decode(
            output_tokens,
            skip_special_tokens=True
        ).strip()

        # -------- CLEAN OUTPUT --------
        if "Answer:" in answer_text:
            answer_text = answer_text.split("Answer:")[-1].strip()

        return answer_text