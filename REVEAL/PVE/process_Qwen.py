import os
from tqdm import tqdm
import gzip
import json
import numpy as np
import torch
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def get_itemmap(dataset):
    dataname = dataset + '.json.gz'
    itemmap = dict()
    itemnum = 1
    for one_interaction in parse(dataname):
        asin = one_interaction['asin']
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemid = itemnum
            itemmap[asin] = itemid
            itemnum += 1
    return itemmap

def read_image_exists(itemmap, image_path):
    parent_folder = image_path
    subfolders = set([f.split('.')[0] for f in os.listdir(parent_folder)])
    has_image = {}
    for asin, itemid in itemmap.items():
        has_image[itemid] = (asin in subfolders)
    return has_image


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET = "Home/"
dataname = "Home"

itemmap = get_itemmap(dataset=f"{DATASET}reviews_{dataname}_5")
has_image = read_image_exists(itemmap, image_path=f"{DATASET}image_text/{dataname}_image")

device = "cuda" if torch.cuda.is_available() else "cpu"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
    quantization_config=bnb_config,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

model.eval()

vision_device = next(model.visual.parameters()).device


image_features_list = []
feat_dim = None
error_item_num = 0

for asin, itemid in tqdm(itemmap.items(), total=len(itemmap)):
    img_path = f"{DATASET}image_text/{dataname}_image/{asin}.png"

    if has_image.get(itemid, False) and os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        if min(w, h) < 28:
            scale = 28 / min(w, h)
            img = img.resize(
                (int(round(w * scale)), int(round(h * scale))),
                Image.BICUBIC
            )

        vis_inp = processor.image_processor(images=img, return_tensors="pt")
        pixel_values = vis_inp["pixel_values"].to(vision_device)
        grid_thw = vis_inp["image_grid_thw"].to(vision_device)

        with torch.no_grad():
            vision_out = model.visual(pixel_values, grid_thw)  # :contentReference[oaicite:2]{index=2}

            if isinstance(vision_out, (tuple, list)):
                vision_out = vision_out[0]
            # vision_out: [B, N, D] or [N, D]
            if vision_out.dim() == 3:
                token_feats = vision_out[0]          # [N, D]
            else:
                token_feats = vision_out              # [N, D]

            # pooling -> [D]
            img_feat = token_feats.mean(dim=0, keepdim=True)  # mean pooling
            img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-12)

            if feat_dim is None:
                feat_dim = img_feat.shape[-1]

            image_features_list.append(img_feat.detach().to("cpu"))

    else:
        error_item_num += 1
        image_features_list.append(None)

print("missing images:", error_item_num)

if feat_dim is None:
    raise RuntimeError("No valid images found, cannot infer embedding dim.")

for i in range(len(image_features_list)):
    if image_features_list[i] is None:
        rand = torch.normal(mean=0.0, std=0.02, size=(1, feat_dim))
        rand = rand / (rand.norm(dim=-1, keepdim=True) + 1e-12)
        image_features_list[i] = rand

image_features_tensor = torch.cat(image_features_list, dim=0)  # [num_items, D]

save_path = os.path.join("Features", DATASET, f"qwen25vl_image_features.pt")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(image_features_tensor, save_path)

print("saved to:", save_path, "shape:", tuple(image_features_tensor.shape))
