import os
import json
from copy import deepcopy
from tqdm import tqdm
from openai import OpenAI



DATASET_ROOT = "../Home"

PROMPT_V1_PATH = os.path.join(DATASET_ROOT, "image_text", "Home_prompt.json")

OUT_PROMPT_V2_PATH = os.path.join(DATASET_ROOT, "image_text", "Home_promptV2_New.json")

PROMPT_OPT_INPUT_PATH = os.path.join("Home_prompt_opt_input_level1.json")

MODEL_NAME = "qwen2.5-vl-7b-instruct"  

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key='',
)




def load_prompts_v1(path: str):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # {asin: prompt_str}


def load_prompt_opt_samples(path: str):
    """
    {
      "asin": "...",
      "item_id": ...,
      "old_prompt": "...",
      "review_summary": "...",
      "rank_feedback": {...},
      "priority": ...
    }
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples



def build_llm_instruction(sample):
    asin = sample["asin"]
    item_id = sample["item_id"]
    old_prompt = sample["old_prompt"]
    review_summary = sample["review_summary"]
    fb = sample["rank_feedback"]
    priority = sample["priority"]

    instr = f"""
You are an expert at designing vision prompts for CLIP-like models used in a recommender system.

We use CLIP + a text prompt to extract image features for items, and then feed them into a sequential recommender.
Your mission is to improve the image prompt so that CLIP focuses on the visual cues that matter most to real users, 
thereby improving the downstream ranking quality.

Below is one item with its current prompt, user review summary (what users repeatedly care about), 
and ranking feedback showing how poorly this item is currently ranked.

[Item Info]
- ASIN: {asin}
- Internal item_id: {item_id}
- Priority score for prompt optimization: {priority:.4f}

[Current Prompt V1]
{old_prompt}

[User Review Summary — what real users care about]
{review_summary}

[Ranking Feedback]
- hit_count: {fb['hit_count']}
- avg_rank: {fb['avg_rank']:.2f}
- top10_rate: {fb['top10_rate']:.2f}
- top20_rate: {fb['top20_rate']:.2f}

Interpretation:
This item appears frequently in user interaction sequences but is still ranked too low. 
This strongly suggests that Prompt V1 does not highlight the right visual cues that reflect what users actually care about.

Your tasks:

1. ANALYSIS (very concise):
   - Based on the review summary, extract the key concerns users repeatedly mention.
   - Compare those concerns with what Prompt V1 visually focuses on.
   - Explain the mismatch: which user-important visual cues (that are visible in the image) are missing or under-emphasized in V1.
   - Use **no more than 3 sentences** (ideally 40–60 words total).
   - Structure:
       Users care about: <list of important user-focused attributes>.
       Prompt V1 focuses on: <what V1 visually emphasizes>.
       Mismatch: <what crucial visually-observable cues are missing, causing poor ranking>.

2. NEW_PROMPT (Prompt V2):
   Design a new CLIP vision prompt that helps CLIP highlight the visual cues most aligned 
   with real user preferences for this item.

   Requirements:
   - Use ONLY visually observable attributes, but **select and phrase them according to user preferences** 
     (e.g., natural look, softness, compactness, shape precision, premium finish, ergonomic form, etc.).
   - Prioritize user-valued cues over generic description. 
     The result should NOT look like an ordinary product caption.
     It should make CLIP encode the item along the dimensions users care about.
   - Remove details irrelevant to user preferences.
   - Keep the prompt concise (ideally within 30 words, max 40).
   - No marketing claims, no usage descriptions, no functional promises—only visual expressions that align with user concerns.

Output format (very important):
You MUST output in plain text, NO markdown, NO code fences, NO JSON.
Use EXACTLY the following two labeled sections:

ANALYSIS:
<1–3 short sentences: what users care about, what V1 misses, why ranking suffers>

NEW_PROMPT:
<Your improved English vision prompt (≤40 words)>
"""
    return instr



def parse_llm_output(assistant_text: str, asin: str, default_prompt_v1: str):
    """

    ANALYSIS:
    ...

    NEW_PROMPT:
    ...
    """
    if not assistant_text:
        print(f"[WARN] Empty LLM output for ASIN={asin}, fallback to V1.")
        return default_prompt_v1, ""

    text = assistant_text.strip()

    upper_text = text.upper()

    idx_new = upper_text.find("NEW_PROMPT:")
    if idx_new == -1:
        print(f"[WARN] 'NEW_PROMPT:' not found for ASIN={asin}, fallback to V1.")
        return default_prompt_v1, ""

    part_analysis = text[:idx_new]
    part_new = text[idx_new + len("NEW_PROMPT:"):]

    analysis = part_analysis.strip()
    upper_ana = analysis.upper()
    idx_ana = upper_ana.find("ANALYSIS:")
    if idx_ana != -1:
        analysis = analysis[idx_ana + len("ANALYSIS:"):].strip()

    new_prompt = part_new.strip()
    if not new_prompt:
        print(f"[WARN] Parsed NEW_PROMPT is empty for ASIN={asin}, fallback to V1.")
        return default_prompt_v1, analysis

    words = new_prompt.split()
    if len(words) > 40:
        new_prompt = " ".join(words[:40])

    return new_prompt, analysis


def optimize_single_prompt_with_feedback(sample, default_prompt_v1: str):

    asin = sample["asin"]
    instr = build_llm_instruction(sample)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that strictly follows the user's instructions and output format.",
        },
        {
            "role": "user",
            "content": instr,
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
        return default_prompt_v1, ""

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

    new_prompt, analysis = parse_llm_output(assistant_text, asin, default_prompt_v1)
    return new_prompt, analysis




def main():
    print(f"[INFO] Loading Prompt V1 from: {PROMPT_V1_PATH}")
    asin2prompt_v1 = load_prompts_v1(PROMPT_V1_PATH)
    print(f"[INFO] Total prompts in V1: {len(asin2prompt_v1)}")

    print(f"[INFO] Loading prompt optimization samples from: {PROMPT_OPT_INPUT_PATH}")
    samples = load_prompt_opt_samples(PROMPT_OPT_INPUT_PATH)
    print(f"[INFO] Total samples for optimization: {len(samples)}")

    asin2prompt_v2 = deepcopy(asin2prompt_v1)

    changed_prompts = {}

    for sample in tqdm(samples, desc="Optimizing prompts with feedback"):
        asin = sample["asin"]
        old_prompt = asin2prompt_v1.get(asin, sample.get("old_prompt", ""))

        if not old_prompt:
            print(f"[WARN] No V1 prompt found for ASIN={asin}, skip.")
            continue
        new_prompt, analysis = optimize_single_prompt_with_feedback(sample, default_prompt_v1=old_prompt)
        asin2prompt_v2[asin] = new_prompt

        if new_prompt.strip() != old_prompt.strip():
            changed_prompts[asin] = {
                "old_prompt": old_prompt,
                "new_prompt": new_prompt,
                "analysis": analysis,
            }

    os.makedirs(os.path.dirname(OUT_PROMPT_V2_PATH), exist_ok=True)
    with open(OUT_PROMPT_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(asin2prompt_v2, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved full PromptV2 to: {OUT_PROMPT_V2_PATH}")
    print(f"[STATS] V2 prompts count (all asin): {len(asin2prompt_v2)}")

    diff_path = OUT_PROMPT_V2_PATH.replace(".json", "_diff_debug.json")
    with open(diff_path, "w", encoding="utf-8") as f:
        json.dump(changed_prompts, f, ensure_ascii=False, indent=2)

    print(f"[STATS] Changed prompts count: {len(changed_prompts)}")
    print(f"[INFO] Diff of changed prompts (with analysis) saved to: {diff_path}")

    print("\n[EXAMPLE] Some changed prompts with analysis:")
    shown = 0
    for asin, v in list(changed_prompts.items()):
        print(f"\nASIN: {asin}")
        print(f"  ANALYSIS: {v.get('analysis', '')}")
        print(f"  OLD: {v['old_prompt']}")
        print(f"  NEW: {v['new_prompt']}")
        shown += 1
        if shown >= 3:
            break

if __name__ == "__main__":
    main()