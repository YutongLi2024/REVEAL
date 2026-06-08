import os
import json
from copy import deepcopy
from tqdm import tqdm

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)


DATASET_ROOT = "../Sports"


PROMPT_V1_PATH = os.path.join(DATASET_ROOT, "image_text", "Sports_prompt.json")


OUT_PROMPT_V2_PATH = os.path.join(DATASET_ROOT, "image_text", "Sports_promptV2_Neg.json")


PROMPT_OPT_INPUT_PATH = os.path.join("Sports_prompt_opt_input_all.json")


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"


MAX_NEW_TOKENS = 192




def init_qwen_model():
    num_gpus = torch.cuda.device_count()
    print("[QWEN] torch.cuda.device_count() =", num_gpus)

    if num_gpus > 0:
        max_memory = {i: "18GiB" for i in range(num_gpus)}
        device_map = "balanced_low_0"
    else:
        max_memory = None
        device_map = "cpu"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        attn_implementation="sdpa",
        quantization_config=bnb_config,
    )

    print("[QWEN] hf_device_map =", getattr(model, "hf_device_map", None))

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    return model, processor



qwen_model, qwen_processor = init_qwen_model()


def qwen_generate_text(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = qwen_processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )

    hf_device_map = getattr(qwen_model, "hf_device_map", None)
    if isinstance(hf_device_map, dict) and len(hf_device_map) > 0:
        first_device_str = list(set(hf_device_map.values()))[0]
        first_device = torch.device(first_device_str)
    else:
        first_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(first_device)

    qwen_model.eval()
    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0] if output_text else ""




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
      "rank_feedback": {
          "hit_count": ...,
          "avg_pos_rank": ...,
          "pos_top10_rate": ...,
          "pos_top20_rate": ...,
          "wrong_hit_topK": ...,
          "wrong_avg_rank_topK": ...,
          "wrong_top10_rate": ...,
          "exposure_count_topK": ...,
          "avg_exposure_rank_topK": ...,
          ...
      },
      "priority": ...,
      "sample_type": "pos" or "neg"
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
    sample_type = sample.get("sample_type", "pos")

    hit_count = fb.get("hit_count", 0)
    avg_pos_rank = fb.get("avg_pos_rank", fb.get("avg_rank", 0.0))
    pos_top10 = fb.get("pos_top10_rate", fb.get("top10_rate", 0.0))
    pos_top20 = fb.get("pos_top20_rate", fb.get("top20_rate", 0.0))

    wrong_hit = fb.get("wrong_hit_topK", 0)
    wrong_avg_rank = fb.get("wrong_avg_rank_topK", 0.0)
    wrong_top10 = fb.get("wrong_top10_rate", 0.0)

    exposure_count = fb.get("exposure_count_topK", 0)
    avg_exposure_rank = fb.get("avg_exposure_rank_topK", 0.0)

    pos_diff = fb.get("pos_difficulty", None)
    neg_over = fb.get("neg_overexposure", None)
    exposure_imp = fb.get("exposure_importance", None)

    if sample_type == "neg":
        optimization_goal = (
            "For this item, the main goal is to REDUCE wrong recommendations.\n"
            "It is frequently pushed into top-K positions when it is NOT the true target.\n"
            "The new vision prompt should make the item visually more specific and less generic, "
            "so that CLIP does not confuse it with other popular items for unrelated users."
        )
    else:
        optimization_goal = (
            "For this item, the main goal is to IMPROVE ranking when it IS the true target.\n"
            "Many real users interact with this item, but it is still ranked too low in those sequences.\n"
            "The new vision prompt should highlight visual cues that match what these users truly care about."
        )

    instr = f"""
You are an expert at designing vision prompts for CLIP-like models used in a recommender system.

We use CLIP + a text prompt to extract image features for items, and then feed them into a sequential recommender.
Your mission is to improve the image prompt so that CLIP focuses on the visual cues that matter most to real users,
thereby improving the downstream ranking quality.

Below is one item with its current prompt, user review summary (what users repeatedly care about),
and ranking feedback (both when it should be recommended and when it is wrongly recommended).

[Item Info]
- ASIN: {asin}
- Internal item_id: {item_id}
- Priority score for prompt optimization: {priority:.4f}
- Sample type for optimization: {sample_type}  (pos = bad ranking as true target; neg = frequent wrong top-K recommendation)

[Current Prompt V1]
{old_prompt}

[User Review Summary — what real users care about]
{review_summary}

[Ranking Feedback when this item IS the true target]
- hit_count (number of test occurrences): {hit_count}
- avg_pos_rank (smaller is better): {avg_pos_rank:.2f}
- pos_top10_rate: {pos_top10:.2f}
- pos_top20_rate: {pos_top20:.2f}

[Ranking Feedback when this item is WRONGLY recommended in top-K]
- wrong_hit_topK (how many times it appears in top-K as a negative item): {wrong_hit}
- wrong_avg_rank_topK (average rank in those wrong recommendations): {wrong_avg_rank:.2f}
- wrong_top10_rate: {wrong_top10:.2f}

[Exposure Summary]
- exposure_count_topK (how many times it appears in the shown top-K list): {exposure_count}
- avg_exposure_rank_topK: {avg_exposure_rank:.2f}

[Derived difficulty scores (if available)]
- pos_difficulty (how hard it is to rank well when it is relevant): {pos_diff}
- neg_overexposure (how severe the wrong recommendations are): {neg_over}
- exposure_importance (how important this item is overall): {exposure_imp}

Optimization goal:
{optimization_goal}

Your tasks:

1. ANALYSIS (very concise):
   - From the review summary, extract the key visual-related concerns users repeatedly mention
     (e.g., natural look, softness, compactness, robust build, ergonomic shape, premium finish, etc.).
   - Compare these concerns with what Prompt V1 visually emphasizes.
   - Explain the mismatch: which user-important visual cues (that are visible in the image) are missing or under-emphasized,
     and how this mismatch is consistent with the ranking feedback above
     (either poor ranking when relevant, or frequent wrong top-K recommendations).
   - Use no more than 3 sentences (ideally 40–60 words total).
   - Structure:
       Users care about: <list of important user-focused attributes>.
       Prompt V1 focuses on: <what V1 visually emphasizes>.
       Mismatch: <which crucial visually observable cues are missing or too generic, and how this leads to the observed ranking problems>.

2. NEW_PROMPT (Prompt V2):
   Design a new CLIP vision prompt that helps CLIP highlight the visual cues most aligned
   with real user preferences for this item, according to the sample_type:

   - If sample_type = "pos":
       Focus on visual aspects that make this item especially attractive for its true users,
       so that CLIP can better separate it from other items and rank it higher in those sequences.

   - If sample_type = "neg":
       Focus on visual aspects that make this item more specific and less generic,
       so that CLIP does not misinterpret it as a common generic item for many unrelated users.
       Emphasize distinct visual traits that narrow down its target user group.

   Requirements for Prompt V2:
   - Use ONLY visually observable attributes, but select and phrase them according to user preferences
     (e.g., softness, compact size, heavy-duty look, rich color palette, sturdy straps, reflective patterns, etc.).
   - Prioritize user-valued cues over generic description.
     The result should NOT look like an ordinary product caption.
   - Remove details irrelevant to user preferences.
   - Keep the prompt concise (ideally within 30 words, max 40).
   - No marketing claims, no usage descriptions, no functional promises—only visual expressions that align with user concerns.

Output format (very important):
You MUST output in plain text, NO markdown, NO code fences, NO JSON.
Use EXACTLY the following two labeled sections:

ANALYSIS:
<1–3 short sentences: what users care about, what V1 misses, how this causes the observed ranking behavior>

NEW_PROMPT:
<Your improved English vision prompt (≤40 words)>
"""
    return instr


def parse_llm_output(assistant_text: str, asin: str, default_prompt_v1: str):
    """
    (new_prompt, analysis)。


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

    try:
        assistant_text = qwen_generate_text(instr, max_new_tokens=MAX_NEW_TOKENS)
    except Exception as e:
        print(f"[ERROR] Qwen generation failed for ASIN={asin}: {e}")
        return default_prompt_v1, ""

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


    for sample in tqdm(samples, desc="Optimizing prompts with feedback (Qwen local)"):
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
                "sample_type": sample.get("sample_type", "pos"),
                "rank_feedback": sample.get("rank_feedback", {}),
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
        print(f"\nASIN: {asin} (sample_type={v.get('sample_type')})")
        print(f"  ANALYSIS: {v.get('analysis', '')}")
        print(f"  OLD: {v['old_prompt']}")
        print(f"  NEW: {v['new_prompt']}")
        shown += 1
        if shown >= 3:
            break


if __name__ == "__main__":
    main()
