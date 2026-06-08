import os
import json
import base64
from tqdm import tqdm
from openai import OpenAI


DATASET_ROOT = "./Beauty"

PROMPT_V1_PATH = os.path.join(DATASET_ROOT, "image_text", "Beauty_prompt.json")
OUT_PROMPT_V2_PATH = os.path.join(DATASET_ROOT, "image_text", "Beauty_promptV2.json")

IMAGE_DIR = os.path.join(DATASET_ROOT, "image_text", "Beauty_image")  
MODEL_NAME = "qwen3-vl-flash"

TEST_NUM = 5

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=''
)


def load_prompts_v1(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # {asin: prompt_str}


def encode_image_to_data_url(img_path: str) -> str:
 
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    ext = os.path.splitext(img_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{b64}"



SYSTEM_PROMPT = """
You are a visual prompt engineer for CLIP-based vision-language models.

Your goal:
Given a product image and an initial human-written prompt, you must REVISE the prompt so that it is a high-quality visual prompt for guiding image feature extraction.

The prompt will be used as the TEXT input to CLIP to guide attention over the image.

Strict requirements:

1. Use English only.
2. Preserve the overall structure of the original prompt:
   - Start with: "A product photo of ..." and, if present, "from brand ...".
   - Include: Product name: "…".
   - Then contain: "Focus on ..." and "Ignore ..." clauses.
   You may slightly rephrase or shorten these parts, but DO NOT remove brand name or product name unless obviously incorrect.

3. Only describe VISUALLY OBSERVABLE attributes from the image:
   - color, shape, pattern, material appearance (e.g., glossy, matte), logo, printed text, layout of elements (palette, jar, bottle, box, brush, etc.).
   - You may also mention the main category (e.g., concealer palette, cream jar, lotion bottle) based on the image.

4. REMOVE or AVOID non-visual or unverifiable information:
   - effects, functions, benefits, usage scenarios, quality, comfort, "brightening", "professional", "best seller", etc.
   - any attributes that you cannot reasonably see from the image.

5. The final prompt should:
   - be concise (ideally within 30 words, but do NOT exceed 40 words),
   - keep the "Focus on ..." phrase clearly describing which regions or aspects of the product the model should pay attention to,
   - keep the "Ignore ..." phrase to down-weight background, faces, hands, unrelated objects.

6. Very important:
   - Assume the brand name and product name are correct labels for this image. Do NOT delete them just because they are not fully visible.
   - However, if the original prompt mentions an object clearly NOT present in the image, you should remove or correct that part.

OUTPUT FORMAT:
Return ONLY a JSON object with one key:
{
  "prompt_v2": "<your revised prompt here>"
}
Do NOT add any other text outside this JSON.
"""


def optimize_single_prompt_with_image(asin: str, prompt_v1: str) -> str:

    img_path = os.path.join(IMAGE_DIR, f"{asin}.png")
    if not os.path.exists(img_path):
        print(f"[WARN] Image not found for ASIN={asin}, keep original prompt.")
        return prompt_v1

    img_data_url = encode_image_to_data_url(img_path)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_data_url
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Here is the current prompt that will be used as a CLIP text input:\n"
                        f"{prompt_v1}\n\n"
                        "Please revise this prompt according to the rules in the system message "
                        "by carefully checking it against the product image. "
                        "Remember to output ONLY a JSON object with the key 'prompt_v2'."
                    ),
                },
            ],
        },
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=False,
        )
    except Exception as e:
        print(f"[ERROR] API call failed for ASIN={asin}: {e}")
        return prompt_v1

    assistant_text = ""
    try:
        msg_content = completion.choices[0].message.content
        if isinstance(msg_content, list):
            text_parts = []
            for part in msg_content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            assistant_text = "\n".join(text_parts)
        else:
            assistant_text = msg_content or ""
        assistant_text = assistant_text.strip()

        data = json.loads(assistant_text)
        prompt_v2 = data.get("prompt_v2", "").strip()
        if not prompt_v2:
            print(f"[WARN] Empty prompt_v2 for ASIN={asin}, fallback to v1.")
            return prompt_v1
        return prompt_v2

    except Exception as e:
        print(f"[WARN] Failed to parse JSON for ASIN={asin}: {e}")
        if assistant_text:
            return assistant_text.splitlines()[0].strip()
        return prompt_v1


def main():
    print(f"[INFO] Loading Prompt V1 from: {PROMPT_V1_PATH}")
    prompts_v1 = load_prompts_v1(PROMPT_V1_PATH)  # {asin: prompt_str}

    asin_list = list(prompts_v1.keys())
    print(f"[INFO] Total prompts in V1: {len(prompts_v1)}")
    print(f"[INFO] Will optimize ALL {len(prompts_v1)} prompts.\n")

    prompts_v2 = {}

    for asin in tqdm(asin_list, desc="Optimizing prompts with Qwen-VL"):
        prompt_v1 = prompts_v1[asin]
        prompt_v2 = optimize_single_prompt_with_image(asin, prompt_v1)
        prompts_v2[asin] = prompt_v2

    os.makedirs(os.path.dirname(OUT_PROMPT_V2_PATH), exist_ok=True)
    with open(OUT_PROMPT_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(prompts_v2, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved full PromptV2 to: {OUT_PROMPT_V2_PATH}")
    print(f"[STATS] V2 prompts count: {len(prompts_v2)}")



if __name__ == "__main__":
    main()
