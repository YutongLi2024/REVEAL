# process_Qwen_prompt.py
import os
import json
import gzip
from tqdm import tqdm
import torch
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info



def parse_gz_json_lines(path: str):
    """Amazon reviews_*.json.gz: each line is a python dict string, can be eval-ed."""
    with gzip.open(path, "r") as g:
        for l in g:
            yield eval(l)


def build_itemmap_from_reviews(reviews_gz_path: str):
    """
    Build asin -> itemid mapping from reviews_*.json.gz
    itemid starts from 1
    """
    itemmap = {}
    itemnum = 1
    for one_interaction in parse_gz_json_lines(reviews_gz_path):
        asin = one_interaction["asin"]
        if asin not in itemmap:
            itemmap[asin] = itemnum
            itemnum += 1
    return itemmap


def list_existing_image_asins(image_dir: str):
    """Return a set of asins that have image files in image_dir."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    asins = set()
    for fn in os.listdir(image_dir):
        # expect {asin}.png (or jpg). handle common suffixes
        base, ext = os.path.splitext(fn)
        if ext.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            asins.add(base)
    return asins


# =========================
# 1) Qwen
# =========================
def find_image_pad_id(tokenizer):
    """
    Try to find the special token used to represent image patch placeholders in input_ids.
    Different versions may use different names.
    """
    candidates = [
        "<|image_pad|>",
        "<|vision_pad|>",
        "<|img_pad|>",
        "<image_pad>",
        "<vision_pad>",
    ]
    for tok in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                return tid, tok
        except Exception:
            pass
    return None, None


def safe_resize_min_side(img: Image.Image, min_side: int = 28):

    w, h = img.size
    if min(w, h) >= min_side:
        return img
    scale = min_side / float(min(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)


def get_lm_device(model):
    """
    Get device for text embeddings (LLM side).
    model.model.embed_tokens is common in Qwen.
    fallback to model.get_input_embeddings().
    """
    try:
        return next(model.model.embed_tokens.parameters()).device
    except Exception:
        emb = model.get_input_embeddings()
        return next(emb.parameters()).device


def get_vision_device(model):
    """Get device for vision tower."""
    return next(model.visual.parameters()).device


def main():
    MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

    DATASET_DIR = "Home/"

    # reviews gzip
    REVIEWS_GZ = os.path.join(DATASET_DIR, "reviews_Home_5.json.gz")

    IMAGE_DIR = os.path.join(DATASET_DIR, "image_text", "Home_image")

    # prompt json (asin -> prompt)
    PROMPT_JSON = os.path.join(DATASET_DIR, "image_text", "Home_promptV2_New.json")

    SAVE_DIR = os.path.join("Features", DATASET_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(SAVE_DIR, "qwen25vl_promptV2_image_features.pt")


    MAX_IMAGE_SIDE_LIMIT = None  
    DEFAULT_PROMPT = "Describe the main product in this image."

    USE_4BIT = True
    # ----------------------------------------

    print("Loading prompts:", PROMPT_JSON)
    with open(PROMPT_JSON, "r") as f:
        prompt_dict = json.load(f)  # asin(str)->prompt(str)

    print("Building itemmap from:", REVIEWS_GZ)
    itemmap = build_itemmap_from_reviews(REVIEWS_GZ)
    print("Total items from reviews:", len(itemmap))

    print("Indexing existing images from:", IMAGE_DIR)
    existing_asins = list_existing_image_asins(IMAGE_DIR)
    print("Total images found:", len(existing_asins))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print("Loading model:", MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    model.eval()

    vision_device = get_vision_device(model)
    lm_device = get_lm_device(model)
    print("vision_device =", vision_device)
    print("lm_device     =", lm_device)

    image_pad_id, image_pad_tok = find_image_pad_id(processor.tokenizer)
    print("image_pad_tok =", image_pad_tok, "image_pad_id =", image_pad_id)
    if image_pad_id is None:
        print("[Warning] Cannot find image pad token id. Will fallback to using visual-only pooling from model.visual output.")
        print("          (This means prompt will NOT affect embeddings.)")

    feats_cpu = [None] * len(itemmap)  # index by (itemid-1)
    feat_dim = None
    missing_cnt = 0


    torch.manual_seed(2025)

    for asin, itemid in tqdm(itemmap.items(), total=len(itemmap)):
        # itemid starts from 1
        out_index = itemid - 1

        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            p = os.path.join(IMAGE_DIR, f"{asin}{ext}")
            if os.path.exists(p):
                img_path = p
                break

        if (asin in existing_asins) and (img_path is not None):
            img = Image.open(img_path).convert("RGB")

            img = safe_resize_min_side(img, min_side=28)

            if MAX_IMAGE_SIDE_LIMIT is not None:
                w, h = img.size
                max_side = max(w, h)
                if max_side > MAX_IMAGE_SIDE_LIMIT:
                    scale = MAX_IMAGE_SIDE_LIMIT / float(max_side)
                    img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)

            prompt = prompt_dict.get(str(asin), DEFAULT_PROMPT)

            if image_pad_id is not None:
                # ============ Prompt-conditioned embedding ============
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},   # PIL.Image
                        {"type": "text", "text": prompt},
                    ],
                }]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                inputs["input_ids"] = inputs["input_ids"].to(lm_device)
                inputs["attention_mask"] = inputs["attention_mask"].to(lm_device)

                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(vision_device)
                if "image_grid_thw" in inputs:
                    inputs["image_grid_thw"] = inputs["image_grid_thw"].to(vision_device)

                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True, return_dict=True)
                    last_hidden = out.hidden_states[-1]  # [B, seq_len, hidden]
                    img_mask = (inputs["input_ids"] == image_pad_id)  # [B, seq_len]
                    if img_mask.sum().item() == 0:
                        pooled = last_hidden.mean(dim=1)  # [B, hidden]
                    else:
                        img_tokens = last_hidden[0][img_mask[0]]  # [n_img_tokens, hidden]
                        pooled = img_tokens.mean(dim=0, keepdim=True)  # [1, hidden]

                    feat = pooled
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)  # L2 norm

                    if feat_dim is None:
                        feat_dim = feat.shape[-1]

                    feats_cpu[out_index] = feat.detach().to("cpu")

            else:
                # ============ Fallback: visual-only embedding (prompt) ============
                # image_processor + model.visual
                vis_inp = processor.image_processor(images=img, return_tensors="pt")
                pixel_values = vis_inp["pixel_values"].to(vision_device)
                grid_thw = vis_inp["image_grid_thw"].to(vision_device)

                with torch.no_grad():
                    vision_out = model.visual(pixel_values, grid_thw)
                    if isinstance(vision_out, (tuple, list)):
                        vision_out = vision_out[0]
                    if vision_out.dim() == 3:
                        token_feats = vision_out[0]   # [N, D]
                    else:
                        token_feats = vision_out      # [N, D]

                    feat = token_feats.mean(dim=0, keepdim=True)
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)

                    if feat_dim is None:
                        feat_dim = feat.shape[-1]

                    feats_cpu[out_index] = feat.detach().to("cpu")

        else:
            missing_cnt += 1
            feats_cpu[out_index] = None

    print("missing images:", missing_cnt)

    if feat_dim is None:
        raise RuntimeError("No valid images found, cannot infer embedding dim.")

    for i in range(len(feats_cpu)):
        if feats_cpu[i] is None:
            rand = torch.normal(mean=0.0, std=0.02, size=(1, feat_dim))
            rand = rand / (rand.norm(dim=-1, keepdim=True) + 1e-12)
            feats_cpu[i] = rand

    feats = torch.cat(feats_cpu, dim=0)  # [num_items, feat_dim]
    torch.save(feats, SAVE_PATH)
    print("Saved:", SAVE_PATH, "shape:", tuple(feats.shape))


if __name__ == "__main__":
    main()
