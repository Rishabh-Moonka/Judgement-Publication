import torch
from transformers import (
    Idefics3ForConditionalGeneration,
    AutoProcessor
)
from PIL import Image


class Idefics3:

    def __init__(self, device="cuda:0", model_path=None):

        self.device = torch.device(device)

        self.model_id = "HuggingFaceM4/Idefics3-8B-Llama3"
        self.model_path = model_path if model_path else self.model_id

        # -------- PROCESSOR --------
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # -------- MODEL --------
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16
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

        # -------- APPLY TEMPLATE --------
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # -------- PREPARE INPUTS --------
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


if __name__ == "__main__":
    model = Idefics3(device="cuda:0")
    image_path = "test.jpg"
    prompt = "What is in the image?"
    print(model.generate(image_path, prompt))