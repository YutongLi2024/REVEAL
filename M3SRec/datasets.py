import random

import torch
from torch.utils.data import Dataset

from utils import neg_sample


class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]
        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4] train: input_ids  target: target_pos

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

            # if len(items) >= 2:
            #     answer = [items[-2]]
            # else:
            #     answer = [0]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )
        return cur_tensors

    def __len__(self):
        return len(self.user_seq)


import json
import os

class UserLLMDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

        attr_file = os.path.join(args.attributes_dir, 'item2attributes.json')
        
        self.item2attr = None
        if os.path.exists(attr_file):
            print(f"[{data_type.upper()}] Loading attributes from {attr_file}...")
            with open(attr_file, 'r') as f:
                self.item2attr = json.load(f)
        else:
            print(f"Warning: {attr_file} not found! Attributes (Brand/Cate) will be all zeros.")

    def get_attributes(self, item_ids):

        brand_seq = []
        cate_seq = []
        
        if self.item2attr is None:
            return [0] * len(item_ids), [0] * len(item_ids)

        for iid in item_ids:
            if iid == 0: 
                brand_seq.append(0)
                cate_seq.append(0)
                continue
            
            if iid < len(self.item2attr):
                attr = self.item2attr[iid]
                if attr is not None:
                    brand_seq.append(attr['brand'])
                    cate_seq.append(attr['cate'])
                else:
                    brand_seq.append(0)
                    cate_seq.append(0)
            else:
                brand_seq.append(0)
                cate_seq.append(0)
        
        return brand_seq, cate_seq


    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]
        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]
        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        input_brand, input_cate = self.get_attributes(input_ids)

        target_brand, target_cate = self.get_attributes(target_pos)

        # 4. Padding
        pad_len = self.max_len - len(input_ids)
        
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg
        
        input_brand = [0] * pad_len + input_brand
        input_cate = [0] * pad_len + input_cate
        
        target_brand = [0] * pad_len + target_brand
        target_cate = [0] * pad_len + target_cate

        # 5. Truncating
        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]
        
        input_brand = input_brand[-self.max_len:]
        input_cate = input_cate[-self.max_len:]
        
        target_brand = target_brand[-self.max_len:]
        target_cate = target_cate[-self.max_len:]

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            # Input Features
            torch.tensor(input_brand, dtype=torch.long),
            torch.tensor(input_cate, dtype=torch.long),
            torch.tensor(target_brand, dtype=torch.long),
            torch.tensor(target_cate, dtype=torch.long)
        )

    def __len__(self):
        return len(self.user_seq)


class HM4SRDataset(SASRecDataset):
    def __init__(self, args, user_seq, user_time_seq, test_neg_items=None, data_type='train'):
        super().__init__(args, user_seq, test_neg_items=test_neg_items, data_type=data_type)
        self.user_time_seq = user_time_seq  

    def __getitem__(self, index):

        base = super().__getitem__(index)

        if self.test_neg_items is not None:
            user_id, input_ids, target_pos, target_neg, answer, sample_negs = base
        else:
            user_id, input_ids, target_pos, target_neg, answer = base

        times = self.user_time_seq[index]
        assert len(times) >= 1

        if self.data_type == "train":
            t_input = times[:-3]
        elif self.data_type == "valid":
            t_input = times[:-2]
        else:
            t_input = times[:-1]

        pad_len = self.max_len - len(t_input)
        t_input = [0] * pad_len + t_input
        t_input = t_input[-self.max_len:]

        timestamp_list = torch.tensor(t_input, dtype=torch.float32)

        if self.test_neg_items is not None:
            return (user_id, input_ids, target_pos, target_neg, answer, sample_negs, timestamp_list)
        else:
            return (user_id, input_ids, target_pos, target_neg, answer, timestamp_list)


