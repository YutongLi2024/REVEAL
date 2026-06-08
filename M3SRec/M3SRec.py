import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm, DistSAEncoder
from modules import LayerNorm, DistSAEncoder


class Gate(nn.Module):
    def __init__(self, input_dim, low_rank):
        super(Gate, self).__init__()
        self.down = nn.Linear(input_dim, input_dim // low_rank)
        self.up = nn.Linear(input_dim // low_rank, input_dim)

    def forward(self, x):
        x = self.down(x)
        x = torch.sigmoid(x)
        x = self.up(x)
        return x


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, low_rank=1):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, int(hidden_size // low_rank))  
        self.fc2 = nn.Linear(int(hidden_size // low_rank), output_size)
        self.relu = nn.GELU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

class M3SRec(nn.Module):
    def __init__(self, args):
        super(M3SRec, self).__init__()
        self.item_mean_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.item_cov_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_mean_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.position_cov_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.user_margins = nn.Embedding(args.num_users, 1)
        self.item_encoder = DistSAEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        low_rank = args.low_rank

        self.attention_layer = nn.Linear(args.hidden_size * 4, args.hidden_size)

        # text
        self.text_mean_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)
        self.text_cov_embeddings = nn.Embedding(args.item_size, args.pretrain_emb_dim, padding_idx=0)

        # image
        self.image_mean_embeddings = nn.Embedding(args.item_size, args.pretrain_img_emb_dim, padding_idx=0)
        self.image_cov_embeddings = nn.Embedding(args.item_size, args.pretrain_img_emb_dim, padding_idx=0)


        self.fc_mean_image = nn.Linear(args.pretrain_img_emb_dim, args.hidden_size)
        self.fc_cov_image = nn.Linear(args.pretrain_img_emb_dim, args.hidden_size)
        self.fc_mean_text = nn.Linear(args.pretrain_emb_dim, args.hidden_size)
        self.fc_cov_text = nn.Linear(args.pretrain_emb_dim, args.hidden_size)

        # Multi-scale Multimodal Fusion Layer
        self.num_shared_experts = args.num_shared_experts
        self.num_specific_experts = args.num_specific_experts

        self.experts_shared_mean = nn.ModuleList([Expert(self.args.hidden_size * 3, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_shared_experts)])
        self.experts_task1_mean = nn.ModuleList([Expert(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_specific_experts)])
        self.experts_task2_mean = nn.ModuleList([Expert(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_specific_experts)])
        self.experts_task3_mean = nn.ModuleList([Expert(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_specific_experts)])

        self.experts_shared_cov = nn.ModuleList([Expert(self.args.hidden_size * 3, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_shared_experts)])
        self.experts_task1_cov = nn.ModuleList([Expert(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_specific_experts)])
        self.experts_task2_cov = nn.ModuleList([Expert(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_specific_experts)])
        self.experts_task3_cov = nn.ModuleList([Expert(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(self.num_specific_experts)])


        # By using a low-rank expert network and gating, the model parameters can be reduced without sacrificing performance.
        self.dnn_share_mean = nn.Sequential(
            nn.Linear(self.args.hidden_size * 3, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_shared_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn_share_cov = nn.Sequential(
            nn.Linear(self.args.hidden_size * 3, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_shared_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn1_mean = nn.Sequential(
            nn.Linear(self.args.hidden_size, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_specific_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn2_mean = nn.Sequential(
            nn.Linear(self.args.hidden_size, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_specific_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn3_mean = nn.Sequential(
            nn.Linear(self.args.hidden_size, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_specific_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn1_cov = nn.Sequential(
            nn.Linear(self.args.hidden_size, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_specific_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn2_cov = nn.Sequential(
            nn.Linear(self.args.hidden_size, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_specific_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.dnn3_cov = nn.Sequential(
            nn.Linear(self.args.hidden_size, int(self.args.hidden_size / low_rank), bias=False),
            nn.GELU(),
            nn.Linear(int(self.args.hidden_size / low_rank), self.num_specific_experts, bias=False),
            nn.Softmax(dim=2)
        )

        self.apply(self.init_weights)

        print("----------start loading multi_modality -----------")
        self.replace_embedding()

    

    def replace_embedding(self):
        text_features_list = torch.load(self.args.text_emb_path)
        image_features_list = torch.load(self.args.image_emb_path)
        self.image_mean_embeddings.weight.data[1:-1, :] = image_features_list
        self.image_cov_embeddings.weight.data[1:-1, :] = image_features_list
        self.text_mean_embeddings.weight.data[1:-1, :] = text_features_list
        self.text_cov_embeddings.weight.data[1:-1, :] = text_features_list

    ## AGM
    @staticmethod
    def grad_scale(x: torch.Tensor, kappa: torch.Tensor):
        """
        x: [B, L, H] or [B, H]
        kappa: [B, 1, 1] or [B, 1] broadcastable
        forward: unchanged
        backward: scaled by kappa
        """
        return x * kappa + x.detach() * (1.0 - kappa)


    def add_position_mean_embedding(self, sequence, modal="full"):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_mean_embeddings(position_ids)

        item_embeddings = self.item_mean_embeddings(sequence)  # (256,100,64)
        item_image_embeddings = self.fc_mean_image(self.image_mean_embeddings(sequence))
        item_text_embeddings = self.fc_mean_text(self.text_mean_embeddings(sequence))

        # ===== AVL =====
        if modal == "img":
            item_text_embeddings = torch.zeros_like(item_text_embeddings)
        elif modal == "txt":
            item_image_embeddings = torch.zeros_like(item_image_embeddings)
        elif modal == "id":
            item_image_embeddings = torch.zeros_like(item_image_embeddings)
            item_text_embeddings  = torch.zeros_like(item_text_embeddings)
        # ============================================

        
        if self.args.is_use_mm:

            if self.args.is_use_cross:
                x = torch.cat([item_embeddings, item_image_embeddings, item_text_embeddings], axis=-1)

                experts_shared_o = [e(x) for e in self.experts_shared_mean]
                experts_shared_o = torch.stack(experts_shared_o)
                experts_shared_o = experts_shared_o.squeeze()

                # gate_share
                selected_s = self.dnn_share_mean(x)
                gate_share_out = torch.einsum('abcd, bca -> bcd', experts_shared_o, selected_s)

                experts_task1_o = [e(item_embeddings + gate_share_out) for e in self.experts_task1_mean]
                experts_task1_o = torch.stack(experts_task1_o)
                experts_task2_o = [e(item_image_embeddings + gate_share_out) for e in self.experts_task2_mean]
                experts_task2_o = torch.stack(experts_task2_o)
                experts_task3_o = [e(item_text_embeddings + gate_share_out) for e in self.experts_task3_mean]
                experts_task3_o = torch.stack(experts_task3_o)

                # experts_shared_o = experts_shared_o.squeeze()
                experts_task1_o = experts_task1_o.squeeze()
                experts_task2_o = experts_task2_o.squeeze()
                experts_task3_o = experts_task3_o.squeeze()

                # gate1
                selected1 = self.dnn1_mean(item_embeddings)
                gate_1_out = torch.einsum('abcd, bca -> bcd', experts_task1_o, selected1)

                # gate2
                selected2 = self.dnn2_mean(item_image_embeddings)
                gate_2_out = torch.einsum('abcd, bca -> bcd', experts_task2_o, selected2)

                # gate3
                selected3 = self.dnn3_mean(item_text_embeddings)
                gate_3_out = torch.einsum('abcd, bca -> bcd', experts_task3_o, selected3)

                # gather
                combined_gate_outputs = torch.cat([gate_1_out, gate_2_out, gate_3_out, gate_share_out], dim=-1)
                attention_scores = self.attention_layer(combined_gate_outputs)
                attention_scores = F.softmax(attention_scores, dim=-1)
                weighted_gate_1_out = gate_1_out[:, 0, :] * attention_scores[:, 0, :]
                weighted_gate_2_out = gate_2_out[:, 0, :] * attention_scores[:, 1, :]
                weighted_gate_3_out = gate_3_out[:, 0, :] * attention_scores[:, 2, :]
                weighted_gate_share_out = gate_share_out[:, 0, :] * attention_scores[:, 3, :]
                task_out = weighted_gate_1_out + weighted_gate_2_out + weighted_gate_3_out + weighted_gate_share_out
                task_out = task_out.unsqueeze(1).expand(-1, 100, -1)  # (batch_size, 100, feature_dim)


                item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings + task_out

            else:
                item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings

        elif self.args.is_use_text:
            item_embeddings = item_embeddings + item_text_embeddings
        elif self.args.is_use_image:
            item_embeddings = item_embeddings + item_image_embeddings


        
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(sequence_emb)

        # return sequence_emb, mm_gate
        return sequence_emb

    def add_position_cov_embedding(self, sequence, modal="full"):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_cov_embeddings(position_ids)

        item_embeddings = self.item_cov_embeddings(sequence)
        item_image_embeddings = self.fc_cov_image(self.image_cov_embeddings(sequence))
        item_text_embeddings = self.fc_cov_text(self.text_cov_embeddings(sequence))

        # ===== AVL =====
        if modal == "img":
            item_text_embeddings = torch.zeros_like(item_text_embeddings)
        elif modal == "txt":
            item_image_embeddings = torch.zeros_like(item_image_embeddings)
        elif modal == "id":
            item_image_embeddings = torch.zeros_like(item_image_embeddings)
            item_text_embeddings  = torch.zeros_like(item_text_embeddings)
        # ============================================

        if self.args.is_use_mm:

            if self.args.is_use_cross:
                # print("use cross")

                x = torch.cat([item_embeddings, item_image_embeddings, item_text_embeddings], axis=-1)

                # share 
                experts_shared_o = [e(x) for e in self.experts_shared_cov]
                experts_shared_o = torch.stack(experts_shared_o)
                experts_shared_o = experts_shared_o.squeeze()

                selected_s = self.dnn_share_cov(x)
                gate_share_out = torch.einsum('abcd, bca -> bcd', experts_shared_o, selected_s)

                # modality
                experts_task1_o = [e(item_embeddings + gate_share_out) for e in self.experts_task1_cov]
                experts_task1_o = torch.stack(experts_task1_o)
                experts_task2_o = [e(item_image_embeddings + gate_share_out) for e in self.experts_task2_cov]
                experts_task2_o = torch.stack(experts_task2_o)
                experts_task3_o = [e(item_text_embeddings + gate_share_out) for e in self.experts_task3_cov]
                experts_task3_o = torch.stack(experts_task3_o)

                experts_task1_o = experts_task1_o.squeeze()
                experts_task2_o = experts_task2_o.squeeze()
                experts_task3_o = experts_task3_o.squeeze()

                # gate1
                selected1 = self.dnn1_cov(item_embeddings)  # (256,100,2)
                gate_1_out = torch.einsum('abcd, bca -> bcd', experts_task1_o, selected1)  # (256,100,64)

                # gate2
                selected2 = self.dnn2_cov(item_image_embeddings)
                gate_2_out = torch.einsum('abcd, bca -> bcd', experts_task2_o, selected2)

                # gate3
                selected3 = self.dnn3_cov(item_text_embeddings)
                gate_3_out = torch.einsum('abcd, bca -> bcd', experts_task3_o, selected3)

                # gather
                combined_gate_outputs = torch.cat([gate_1_out, gate_2_out, gate_3_out, gate_share_out], dim=-1)

                attention_scores = self.attention_layer(combined_gate_outputs)

                attention_scores = F.softmax(attention_scores, dim=-1)

                weighted_gate_1_out = gate_1_out[:, 0, :] * attention_scores[:, 0, :]
                weighted_gate_2_out = gate_2_out[:, 0, :] * attention_scores[:, 1, :]
                weighted_gate_3_out = gate_3_out[:, 0, :] * attention_scores[:, 2, :]
                weighted_gate_share_out = gate_share_out[:, 0, :] * attention_scores[:, 3, :]
                task_out = weighted_gate_1_out + weighted_gate_2_out + weighted_gate_3_out + weighted_gate_share_out
                task_out = task_out.unsqueeze(1).expand(-1, 100, -1)  # (batch_size, 100, feature_dim)

                item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings + task_out                                     

            else:
                item_embeddings = item_embeddings + item_image_embeddings + item_text_embeddings

        elif self.args.is_use_text:
            item_embeddings = item_embeddings + item_text_embeddings
        elif self.args.is_use_image:
            item_embeddings = item_embeddings + item_image_embeddings
        

        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(self.dropout(sequence_emb)) + 1

        # return sequence_emb, mm_gate
        return sequence_emb

    def finetune(self, input_ids, user_ids, modal="full"):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * (-2 ** 32 + 1)

        mean_sequence_emb = self.add_position_mean_embedding(input_ids, modal=modal)
        cov_sequence_emb = self.add_position_cov_embedding(input_ids, modal=modal)


        item_encoded_layers = self.item_encoder(mean_sequence_emb,
                                                cov_sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        mean_sequence_output, cov_sequence_output, att_scores = item_encoded_layers[-1]

        margins = self.user_margins(user_ids)

        return mean_sequence_output, cov_sequence_output, att_scores, margins

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.01, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class DistMeanSAModel(M3SRec):
    def __init__(self, args):
        super(DistMeanSAModel, self).__init__(args)
        self.item_encoder = M3SRec(args)






