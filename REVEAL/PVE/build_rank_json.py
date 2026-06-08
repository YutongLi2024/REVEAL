## align_rank_with_asin.py & build_json.py

import json
import os
import math
import numpy as np


BASE_DIR = ".."                 
DATA_NAME = "Beauty"            
DATA_DIR = os.path.join(BASE_DIR, DATA_NAME)

IMAP_PATH = os.path.join(DATA_DIR, "imap.json")
RANK_STATS_RAW_PATH = os.path.join(
    DATA_DIR, f"{DATA_NAME}_prompt_item_rank.json"
)  # item_id -> rank_stats（dict）

ALIGNED_RANK_PATH = os.path.join( f"{DATA_NAME}_prompt_item_rank_asin.json")  

REVIEW_PATH = os.path.join(DATA_DIR, f"{DATA_NAME}_review.json")
PROMPT_PATH = os.path.join(DATA_DIR, "image_text", f"{DATA_NAME}_prompt.json")

OUT_JSONL_PATH = os.path.join(f"{DATA_NAME}_prompt_opt_input_all.json")

USE_TOP_PERCENTILE = True
TOP_PERCENTILE = 75  # 75 → top 25%



def load_imap(imap_path):
    with open(imap_path, "r", encoding="utf-8") as f:
        asin2id = json.load(f)
    id2asin = {int(v): str(k) for k, v in asin2id.items()}
    return id2asin


def load_rank_stats_raw(rank_path):
    with open(rank_path, "r", encoding="utf-8") as f:
        rank_stats = json.load(f)
    return {int(k): v for k, v in rank_stats.items()}


def align_rank_with_asin(imap_path, rank_path, output_path=None):

    id2asin = load_imap(imap_path)
    rank_stats = load_rank_stats_raw(rank_path)

    aligned_items = []
    not_found = []

    for item_id, stats in rank_stats.items():
        asin = id2asin.get(item_id, None)
        if asin is None:
            not_found.append(item_id)
            continue
        aligned_items.append(
            {
                "item_id": item_id,
                "asin": asin,
                **stats,
            }
        )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aligned_items, f, ensure_ascii=False, indent=2)

    print(f"{len(aligned_items)} ")
    if not_found:
        print(
            f" {len(not_found)} item_id  imap.json not found, examples: ",
            not_found[:10],
        )

    return aligned_items


# ======================
# 2. Priority
# ======================

def compute_priority_components(rec, max_pos_rank=100.0, topK_neg=10):

    hit = rec.get("hit_count", 0)
    avg_pos_rank = rec.get("avg_pos_rank", 0.0)
    pos_top10 = rec.get("pos_top10_rate", 0.0)
    pos_top20 = rec.get("pos_top20_rate", 0.0)

    if hit > 0:
        norm_rank = float(avg_pos_rank) / float(max_pos_rank)
        miss10 = 1.0 - float(pos_top10)
        miss20 = 1.0 - float(pos_top20)
        ranking_difficulty = norm_rank + 0.5 * miss10 + 0.5 * miss20
        importance = math.log(1.0 + float(hit))
        pos_difficulty = importance * ranking_difficulty
    else:
        pos_difficulty = 0.0

    wrong_hit = rec.get("wrong_hit_topK", 0)
    wrong_avg_rank = rec.get("wrong_avg_rank_topK", 0.0)
    wrong_top10_rate = rec.get("wrong_top10_rate", 0.0)

    if wrong_hit > 0:
        danger_rank = (topK_neg + 1.0 - float(wrong_avg_rank)) / float(topK_neg)
        danger_rank = max(0.0, min(1.0, danger_rank))
        wrong_importance = math.log(1.0 + float(wrong_hit))
        neg_overexposure = wrong_importance * (
            0.5 * danger_rank + 0.5 * wrong_top10_rate
        )
    else:
        neg_overexposure = 0.0

    exposure_count = rec.get("exposure_count_topK", 0)
    # avg_exposure_rank = rec.get("avg_exposure_rank_topK", 0.0)  
    if exposure_count > 0:
        exposure_importance = math.log(1.0 + float(exposure_count))
    else:
        exposure_importance = 0.0

    return pos_difficulty, neg_overexposure, exposure_importance


def compute_priority_scores(rank_items):
    """
    priority_total = exposure_importance * (α * pos_difficulty_norm + β * neg_overexposure_norm)

    """
    if not rank_items:
        return rank_items, []

    max_pos_rank = 0.0
    for rec in rank_items:
        r = rec.get("avg_pos_rank", 0.0)
        if r > max_pos_rank:
            max_pos_rank = r
    if max_pos_rank <= 0:
        max_pos_rank = 100.0

    pos_list, neg_list, exp_list = [], [], []

    for rec in rank_items:
        pos_dif, neg_over, exp_imp = compute_priority_components(
            rec, max_pos_rank=max_pos_rank, topK_neg=10
        )
        rec["pos_difficulty"] = pos_dif
        rec["neg_overexposure"] = neg_over
        rec["exposure_importance"] = exp_imp

        pos_list.append(pos_dif)
        neg_list.append(neg_over)
        exp_list.append(exp_imp)

    def safe_norm(x, max_x):
        if max_x <= 0:
            return 0.0
        return x / max_x

    max_pos = max(pos_list) if pos_list else 1.0
    max_neg = max(neg_list) if neg_list else 1.0

    scores = []
    for rec in rank_items:
        pos_dif = rec["pos_difficulty"]
        neg_over = rec["neg_overexposure"]
        exp_imp = rec["exposure_importance"]

        pos_norm = safe_norm(pos_dif, max_pos)
        neg_norm = safe_norm(neg_over, max_neg)

        alpha = 1.0
        beta = 1.0

        base_score = alpha * pos_norm + beta * neg_norm
        priority = exp_imp * base_score

        rec["priority"] = float(priority)
        scores.append(priority)

        if pos_dif >= neg_over:
            rec["sample_type"] = "pos"
        else:
            rec["sample_type"] = "neg"

    return rank_items, scores


# ======================
# 3.  review / prompt
# ======================

def load_review(path):
    """ {DATA_NAME}_review.json: asin -> {num_reviews, used_reviews, text}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_asin2prompt(path):
    """ {DATA_NAME}_prompt.json: asin -> PromptV1 string """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ======================
# 4.   priority + JSONL
# ======================

def main():
    # 1)  rank_stats（{DATA_NAME}_all_item_rank.json）
    if not os.path.exists(IMAP_PATH):
        raise FileNotFoundError(f"imap.json: {IMAP_PATH}")
    if not os.path.exists(RANK_STATS_RAW_PATH):
        raise FileNotFoundError(
            f" rank_stats json: {RANK_STATS_RAW_PATH}"
        )

    print(f"[INFO]  imap  item_id -> asin ...")
    rank_items = align_rank_with_asin(IMAP_PATH, RANK_STATS_RAW_PATH, ALIGNED_RANK_PATH)
    print(f"[INFO]  {len(rank_items)}  item: {ALIGNED_RANK_PATH}")

    print("[INFO] priority  sample_type ...")
    rank_items, scores = compute_priority_scores(rank_items)
    if not scores:
        print("[ERROR]")
        return

    if USE_TOP_PERCENTILE:
        threshold = float(np.percentile(scores, TOP_PERCENTILE))
        print(
            f"[INFO] priority  (>= {TOP_PERCENTILE}  → top {100 - TOP_PERCENTILE}%): {threshold:.4f}"
        )
        selected_items = [rec for rec in rank_items if rec["priority"] >= threshold]
    else:
        threshold = None
        selected_items = rank_items
        print(
            f"[INFO]  priority  {len(selected_items)}  item  Prompt optimization."
        )

    num_pos_items = sum(
        1 for rec in selected_items if rec.get("hit_count", 0) > 0
    )
    num_neg_items = sum(
        1 for rec in selected_items if rec.get("wrong_hit_topK", 0) > 0
    )
    print(
        f"[DEBUG] selected_items: total={len(selected_items)}, "
        f"with_pos={num_pos_items}, with_neg={num_neg_items}"
    )

    # 3) review / prompt
    if not os.path.exists(REVIEW_PATH):
        raise FileNotFoundError(f"{DATA_NAME}_review.json: {REVIEW_PATH}")
    if not os.path.exists(PROMPT_PATH):
        raise FileNotFoundError(f"{DATA_NAME}_prompt.json: {PROMPT_PATH}")

    print(f" {REVIEW_PATH}")
    asin2review = load_review(REVIEW_PATH)
    print(f" PromptV1: {PROMPT_PATH}")
    asin2prompt = load_asin2prompt(PROMPT_PATH)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_JSONL_PATH, "w", encoding="utf-8") as fout:
        for rec in selected_items:
            asin = rec["asin"]
            item_id = rec["item_id"]

            hit_count = rec.get("hit_count", 0)
            avg_pos_rank = rec.get("avg_pos_rank", 0.0)
            pos_top10_rate = rec.get("pos_top10_rate", 0.0)
            pos_top20_rate = rec.get("pos_top20_rate", 0.0)

            wrong_hit_topK = rec.get("wrong_hit_topK", 0)
            wrong_avg_rank_topK = rec.get("wrong_avg_rank_topK", 0.0)
            wrong_top10_rate = rec.get("wrong_top10_rate", 0.0)

            exposure_count_topK = rec.get("exposure_count_topK", 0)
            avg_exposure_rank_topK = rec.get("avg_exposure_rank_topK", 0.0)

            priority = rec["priority"]
            sample_type = rec.get("sample_type", "pos")

            old_prompt = asin2prompt.get(asin, "")
            review_info = asin2review.get(asin, {})
            review_text = review_info.get("text", "")

            sample = {
                "asin": asin,
                "item_id": item_id,
                "old_prompt": old_prompt,
                "review_summary": review_text,
                "rank_feedback": {
                    "hit_count": hit_count,
                    "avg_pos_rank": avg_pos_rank,
                    "pos_top10_rate": pos_top10_rate,
                    "pos_top20_rate": pos_top20_rate,
                    "wrong_hit_topK": wrong_hit_topK,
                    "wrong_avg_rank_topK": wrong_avg_rank_topK,
                    "wrong_top10_rate": wrong_top10_rate,
                    "exposure_count_topK": exposure_count_topK,
                    "avg_exposure_rank_topK": avg_exposure_rank_topK,
                    "pos_difficulty": rec.get("pos_difficulty", 0.0),
                    "neg_overexposure": rec.get("neg_overexposure", 0.0),
                    "exposure_importance": rec.get("exposure_importance", 0.0),
                },
                "priority": priority,
                "sample_type": sample_type,
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f" Prompt: {OUT_JSONL_PATH}")

    print("\n[EXAMPLE] LLM ")
    with open(OUT_JSONL_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if first_line:
        example = json.loads(first_line)
        print(json.dumps(example, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
