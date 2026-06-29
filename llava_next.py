import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image


class LlavaNext:
    def __init__(self, device="cuda:1", model_path=None):
        self.device = torch.device(device)

        if model_path is None:
            model_path = "llava-hf/llava-v1.6-mistral-7b-hf"

        self.processor = AutoProcessor.from_pretrained(model_path)

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)

        self.model.eval()

    def generate(self, image_path, prompt, max_new_tokens=256):

        if isinstance(prompt, dict):
            sys_prompt = prompt.get("system prompt", "")
            user_prompt = prompt.get("user prompt", "")
        else:
            sys_prompt = ""
            user_prompt = prompt

        messages = []

        if sys_prompt:
            messages.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": sys_prompt}
                ]
            })

        user_content = []

        if image_path is not None:
            user_content.append({
                "type": "image",
                "image": image_path
            })

        user_content.append({
            "type": "text",
            "text": user_prompt
        })

        messages.append({
            "role": "user",
            "content": user_content
        })

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
            images = [image]
        else:
            images = None

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        return output[0]