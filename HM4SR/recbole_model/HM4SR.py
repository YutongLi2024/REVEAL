import torch
from torch import nn
from torch.nn import functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
import pickle
import math
import random


class HM4SR(SequentialRecommender):
    def __init__(self, config, dataset):
        super(HM4SR, self).__init__(config, dataset)
        self.data_name = config["dataset"].split('/')[-1]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.temperature = config["temperature"]
        self.phcl_temperature = config["phcl_temperature"]
        self.phcl_weight = config["phcl_weight"]
        self.beta = config["beta"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.item_seq = TransformerEncoder(
            n_layers=self.n_layers, n_heads=self.n_heads,
            hidden_size=self.hidden_size, inner_size=self.inner_size, hidden_dropout_prob=self.hidden_dropout_prob, attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)
        self.txt_seq = TransformerEncoder(
            n_layers=self.n_layers, n_heads=self.n_heads,
            hidden_size=self.hidden_size, inner_size=self.inner_size, hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)
        self.img_seq = TransformerEncoder(
            n_layers=self.n_layers, n_heads=self.n_heads,
            hidden_size=self.hidden_size, inner_size=self.inner_size, hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)

        self.item_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.txt_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.img_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # 增加模态映射
        self.txt_projection = nn.Linear(768, self.hidden_size)
        self.img_projection = nn.Linear(768, self.hidden_size)

        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss()

        # 增加时序信息
        self.time_moe = Temporal_MoE_C(config)

        self.apply(self._init_weights)

        # 增加模态嵌入
        self.txt_embedding = nn.Embedding.from_pretrained(torch.load(f'./dataset/{self.data_name}/txt_emb.pt'))
        self.img_embedding = nn.Embedding.from_pretrained(torch.load(f'./dataset/{self.data_name}/img_emb.pt'))
        # 增加属性类别预测任务
        cat_emb = torch.load(f'./dataset/{self.data_name}/cat.pt').float()
        self.cat_embedding = nn.Embedding.from_pretrained(cat_emb)
        self.cat_linear = nn.Linear(3 * self.hidden_size, cat_emb.shape[-1])
        self.cat_criterion = nn.BCEWithLogitsLoss()
        # 增加初始MoE
        self.start_moe = Align_MoE(config)
        # 增加placeholder编码器
        self.placeholder_txt = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.placeholder_img = nn.Linear(2*self.hidden_size, self.hidden_size)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_idx, seq_length, timestamp=None):
        # 嵌入映射
        item_emb = self.item_embedding(input_idx)
        txt_emb = self.txt_projection(self.txt_embedding(input_idx))
        img_emb = self.img_projection(self.img_embedding(input_idx))
        # 位置嵌入
        id_pos_emb = self.position_embedding.weight[:input_idx.shape[1]]
        id_pos_emb = id_pos_emb.unsqueeze(0).repeat(item_emb.shape[0], 1, 1)
        item_emb += id_pos_emb
        txt_emb += id_pos_emb
        img_emb += id_pos_emb
        ### 添加MoE ###
        align_info = self.start_moe(torch.cat([item_emb, txt_emb, img_emb], dim=-1))
        item_emb += align_info[0]
        txt_emb += align_info[1]
        img_emb += align_info[2]
        ### 添加时序MoE ###
        item_emb, txt_emb, img_emb = self.time_moe(torch.cat([item_emb, txt_emb, img_emb], dim=-1), timestamp)
        # 层正则化+dropout
        item_emb_o = self.dropout(self.item_ln(item_emb))
        txt_emb_o = self.dropout(self.txt_ln(txt_emb))
        img_emb_o = self.dropout(self.img_ln(img_emb))
        # 序列编码
        extended_attention_mask = self.get_attention_mask(input_idx)
        item_seq_full = self.item_seq(item_emb_o, extended_attention_mask, output_all_encoded_layers=True)[-1]
        txt_seq_full = self.txt_seq(txt_emb_o, extended_attention_mask, output_all_encoded_layers=True)[-1]
        img_seq_full = self.img_seq(img_emb_o, extended_attention_mask, output_all_encoded_layers=True)[-1]
        item_seq = self.gather_indexes(item_seq_full, seq_length - 1)
        txt_seq = self.gather_indexes(txt_seq_full, seq_length - 1)
        img_seq = self.gather_indexes(img_seq_full, seq_length - 1)
        # 预测
        item_emb_full = self.item_embedding.weight
        txt_emb_full = self.txt_projection(self.txt_embedding.weight)
        img_emb_full = self.img_projection(self.img_embedding.weight)

        item_score = torch.matmul(item_seq, item_emb_full.transpose(0, 1))
        txt_score = torch.matmul(txt_seq, txt_emb_full.transpose(0, 1))
        img_score = torch.matmul(img_seq, img_emb_full.transpose(0, 1))
        score = item_score + txt_score + img_score
        return [item_emb, txt_emb, img_emb], [item_seq, txt_seq, img_seq], score

    def calculate_loss(self, interaction):
        item_idx = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        timestamp = interaction['timestamp_list']
        item_emb_seq, seq_vectors, score = self.forward(item_idx, item_seq_len, timestamp)
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(score, pos_items)
        return loss + self.IDCL(seq_vectors[0], interaction) + self.CP(item_idx) + self.PCL(interaction, item_emb_seq, seq_vectors)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        timestamp = interaction['timestamp_list']
        _, _, scores = self.forward(item_seq, item_seq_len, timestamp)
        return scores[:, test_item]

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        timestamp = interaction['timestamp_list']
        _, _, score = self.forward(item_seq, item_seq_len, timestamp)
        return score

    def IDCL(self, seq_pre, interaction):
        # from UniSRec
        seq_output = F.normalize(seq_pre, dim=1)
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))
        pos_items_emb = self.item_embedding(pos_id)
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def CP(self, input_idx, padding_idx=0):
        item_list = input_idx.flatten()
        nonzero_idx = torch.where(input_idx != padding_idx) #
        # 嵌入映射
        item_emb = self.item_embedding(item_list)
        txt_emb = self.txt_projection(self.txt_embedding(item_list))
        img_emb = self.img_projection(self.img_embedding(item_list))
        # 预测类别
        item_attribute_score = self.cat_linear(torch.cat([item_emb, txt_emb, img_emb], dim=-1))
        # 获取答案类别
        item_attribute_target = self.cat_embedding(item_list)
        # 计算损失
        attr_loss = self.cat_criterion(item_attribute_score[nonzero_idx], item_attribute_target[nonzero_idx])
        return attr_loss

    def seq2seq_contrastive(self, seq_1, seq_2, same_pos_id):
        seq_1 = F.normalize(seq_1, dim=1)
        seq_2 = F.normalize(seq_2, dim=1)

        pos_logits = (seq_1 * seq_2).sum(dim=1) / self.phcl_temperature
        pos_logits = torch.exp(pos_logits)
        neg_logits = torch.matmul(seq_1, seq_2.transpose(0, 1)) / self.phcl_temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device),neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean() * self.phcl_weight

    def PCL(self, interaction, item_emb_seq, seq_embs):
        beta = self.beta
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        timestamp = interaction['timestamp_list']
        # 增强部分
        num_mask = torch.floor(item_seq_len * beta).long().tolist()
        masked_item_seq = item_seq.cpu().detach().numpy().copy()
        for i in range(item_seq.shape[0]):
            mask_index = random.sample(range(item_seq_len[i]), k=num_mask[i])
            masked_item_seq[i, mask_index] = -1
        item_seq_aug = torch.tensor(masked_item_seq, dtype=torch.long, device=item_seq.device)
        # 占位符替换物品
        id_embs, txt_embs, img_embs = item_emb_seq[0], item_emb_seq[1], item_emb_seq[2]
        time_embedding = self.time_moe.get_time_embedding(timestamp)
        placeholder_mask = (item_seq_aug == -1).unsqueeze(2)
        txt_embs_aug = txt_embs.masked_fill(placeholder_mask, 0.0)
        txt_placeholder = self.placeholder_txt(time_embedding).masked_fill(~placeholder_mask, 0.0)
        txt_embs_aug += txt_placeholder
        img_embs_aug = img_embs.masked_fill(placeholder_mask, 0.0)
        img_placeholder = self.placeholder_img(time_embedding).masked_fill(~placeholder_mask, 0.0)
        img_embs_aug += img_placeholder
        # 增强表征计算
        txt_embs_aug = self.dropout(self.txt_ln(txt_embs_aug))
        img_embs_aug = self.dropout(self.img_ln(img_embs_aug))
        extended_attention_mask = self.get_attention_mask(item_seq)
        txt_seq_full = self.txt_seq(txt_embs_aug, extended_attention_mask, output_all_encoded_layers=True)[-1]
        img_seq_full = self.img_seq(img_embs_aug, extended_attention_mask, output_all_encoded_layers=True)[-1]
        txt_seq = self.gather_indexes(txt_seq_full, item_seq_len - 1)
        img_seq = self.gather_indexes(img_seq_full, item_seq_len - 1)
        # 对比学习计算
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(item_seq.shape[0], dtype=torch.bool, device=item_seq.device))
        txt_loss, img_loss = self.seq2seq_contrastive(seq_embs[1], txt_seq, same_pos_id), self.seq2seq_contrastive(seq_embs[2], img_seq, same_pos_id)
        return (txt_loss + img_loss) / 2


class Align_MoE(nn.Module):
    def __init__(self, config):
        super(Align_MoE, self).__init__()
        self.expert_num = config["start_expert_num"]
        self.hidden_size = int(config["hidden_size"])
        self.gate_selection = config["start_gate_selection"]
        self.gate_txt = nn.Linear(self.hidden_size, self.expert_num)
        self.gate_img = nn.Linear(self.hidden_size, self.expert_num)
        self.gate_id = nn.Linear(self.hidden_size, self.expert_num)
        self.expert = nn.ModuleList([nn.Linear(self.hidden_size * 3, self.hidden_size * 3) for _ in range(self.expert_num)])  # 先实现最简单的专家网络
        self.weight = nn.Parameter(torch.tensor(config["initializer_weight"]).to('cuda'), requires_grad=True)

    def forward(self, vector):
        # 先只实现softmax
        output = None
        if self.gate_selection == 'softmax':
            expert_output = []
            for i in range(self.expert_num):
                expert_output.append(self.expert[i](vector).unsqueeze(2))
            expert_output = torch.cat(expert_output, dim=2)
            output = []
            output.append(self.weight[0] * torch.sum(expert_output[:,:,:,:self.hidden_size] * F.softmax(self.gate_id(vector[:,:,:self.hidden_size]), dim=-1).unsqueeze(3), dim=2))
            output.append(self.weight[1] * torch.sum(expert_output[:,:,:, self.hidden_size:2 * self.hidden_size] * F.softmax(self.gate_txt(vector[:,:,self.hidden_size:2 * self.hidden_size]), dim=-1).unsqueeze(3), dim=2))
            output.append(self.weight[2] * torch.sum(expert_output[:,:,:,2 * self.hidden_size:] * F.softmax(self.gate_img(vector[:,:,2 * self.hidden_size:]), dim=-1).unsqueeze(3), dim=2))
        return output


class Temporal_MoE_C(nn.Module):
    def __init__(self, config):
        super(Temporal_MoE_C, self).__init__()
        self.data_name = config["dataset"].split('/')[-1]
        self.interval_scale = config["interval_scale"]
        self.hidden_size = int(config["hidden_size"])
        self.expert_num = config["temporal_expert_num"]
        self.gate_selection = config["temporal_gate_selection"]
        self.gate = nn.Linear(2 * self.hidden_size, self.expert_num)
        self.absolute_w = nn.Linear(1, self.hidden_size)
        self.absolute_m = nn.Linear(self.hidden_size, self.hidden_size)
        self.time_embedding = nn.Embedding(int(self.interval_scale * self.get_interval_num()) + 1, self.hidden_size)

        self.expert = [nn.Parameter(torch.Tensor(1, self.hidden_size * 3).to('cuda'), requires_grad=True) for _ in range(self.expert_num)]
        for i in range(self.expert_num):
            nn.init.normal_(self.expert[i], std=0.1)

    def get_interval_num(self):
        with open(f'./dataset/{self.data_name}/interval_num', 'rb') as f: return pickle.load(f)

    def get_minmax_day(self):
        with open(f'./dataset/{self.data_name}/minmax_num', 'rb') as f: return pickle.load(f)

    def get_time_embedding(self, timestamp):
        absolute_embedding = torch.cos(self.freq_enhance_ab(self.absolute_w(timestamp.unsqueeze(2))))
        interval_first = torch.zeros((timestamp.shape[0], 1)).long().to('cuda')
        interval = torch.log2(timestamp[:, 1:] - timestamp[:, :-1] + 1)
        interval_index = torch.floor(self.interval_scale * interval).long()
        interval_index = torch.cat([interval_first, interval_index], dim=-1)
        interval_embedding = self.time_embedding(interval_index)
        return torch.cat([interval_embedding, absolute_embedding], dim=-1)

    def freq_enhance_ab(self, timestamp):
        freq = 10000
        freq_seq = torch.arange(0, self.hidden_size, 1.0, dtype=torch.float).to('cuda')
        inv_freq = 1 / torch.pow(freq, (freq_seq / self.hidden_size)).view(1, -1) # shape = (64)
        return timestamp * inv_freq

    def forward(self, vector, timestamp):
        # 先只实现softmax
        expert_proba = None
        absolute_embedding = torch.cos(self.freq_enhance_ab(self.absolute_w(timestamp.unsqueeze(2))))
        interval_first = torch.zeros((vector.shape[0], 1)).long().to('cuda')
        interval = torch.log2(timestamp[:, 1:] - timestamp[:, :-1] + 1)
        interval_index = torch.floor(self.interval_scale * interval).long()
        interval_index = torch.cat([interval_first, interval_index], dim=-1)
        interval_embedding = self.time_embedding(interval_index)
        route = F.softmax(self.gate(torch.cat([interval_embedding, absolute_embedding], dim=-1)), dim=-1)
        if self.gate_selection == 'softmax':
            expert_output = []
            for i in range(self.expert_num):
                expert_output.append((vector * self.expert[i]).unsqueeze(2))
            expert_output = torch.cat(expert_output, dim=2)
            expert_proba = torch.sum(expert_output * route.unsqueeze(3), dim=2)
        return expert_proba[:, :, :self.hidden_size], expert_proba[:, :, self.hidden_size: 2 * self.hidden_size], expert_proba[:, :, 2 * self.hidden_size:]