# -*- coding: utf-8 -*-
import numpy as np
import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric, cal_mrr
from modules import wasserstein_distance, kl_distance, wasserstein_distance_matmul
from M3SRec.M3SRec import STOSA
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import math


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]), flush=True)
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix, flush=True)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix), None

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        recall_dict_list = []
        ndcg_dict_list = []
        
        # for k in [10, 20]:
        for k in [10, 20, 30, 40]:
            recall_result, recall_dict_k = recall_at_k(answers, pred_list, k)
            recall.append(recall_result)
            recall_dict_list.append(recall_dict_k)
            
            ndcg_result, ndcg_dict_k = ndcg_k(answers, pred_list, k)
            ndcg.append(ndcg_result)
            ndcg_dict_list.append(ndcg_dict_k)
        
        post_fix = {
            "Epoch": epoch,
            "Recall@10": f"{recall[0]:.4f}", "NDCG@10": f"{ndcg[0]:.4f}",
            "Recall@20": f"{recall[1]:.4f}", "NDCG@20": f"{ndcg[1]:.4f}",
            "Recall@30": f"{recall[2]:.4f}", "NDCG@30": f"{ndcg[2]:.4f}",
            "Recall@40": f"{recall[3]:.4f}", "NDCG@40": f"{ndcg[3]:.4f}",
        }
        
        print(post_fix, flush=True)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')

        item_rank_stats = self.aggregate_item_rank_stats(answers, pred_list)

        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[3], ndcg[3]], str(post_fix), [recall_dict_list, ndcg_dict_list], item_rank_stats


    def get_pos_items_ranks(self, batch_pred_lists, answers):
        num_users = len(batch_pred_lists)
        batch_pos_ranks = defaultdict(list)
        for i in range(num_users):
            pred_list = batch_pred_lists[i]
            true_set = set(answers[i])
            for ind, pred_item in enumerate(pred_list):
                if pred_item in true_set:
                    batch_pos_ranks[pred_item].append(ind+1)
        return batch_pos_ranks

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location='cuda:0'))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc

    def predict_sample(self, seq_out, test_neg_sample):
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


    def aggregate_item_rank_stats(self, answers, pred_list,
                                  topK_neg=10, topK_exposure=50):


        batch_pos_ranks = self.get_pos_items_ranks(pred_list, answers)

        item_rank_stats = {}

        for item_id, ranks in batch_pos_ranks.items():
            if len(ranks) == 0:
                continue
            hit_count = len(ranks)
            avg_rank = float(np.mean(ranks))
            pos_top10_rate = float(np.mean([1.0 if r <= 10 else 0.0 for r in ranks]))
            pos_top20_rate = float(np.mean([1.0 if r <= 20 else 0.0 for r in ranks]))

            item_rank_stats[int(item_id)] = {
                "hit_count": hit_count,
                "avg_pos_rank": avg_rank,
                "pos_top10_rate": pos_top10_rate,
                "pos_top20_rate": pos_top20_rate,
            }

        batch_neg_ranks = self.get_neg_items_ranks(pred_list, answers, topK=topK_neg)

        for item_id, ranks in batch_neg_ranks.items():
            if len(ranks) == 0:
                continue
            wrong_hit = len(ranks)
            wrong_avg_rank = float(np.mean(ranks))
            wrong_top10_rate = float(np.mean([1.0 if r <= 10 else 0.0 for r in ranks]))

            int_id = int(item_id)
            if int_id not in item_rank_stats:
                item_rank_stats[int_id] = {
                    "hit_count": 0,
                    "avg_pos_rank": 0.0,
                    "pos_top10_rate": 0.0,
                    "pos_top20_rate": 0.0,
                }

            item_rank_stats[int_id]["wrong_hit_topK"] = wrong_hit
            item_rank_stats[int_id]["wrong_avg_rank_topK"] = wrong_avg_rank
            item_rank_stats[int_id]["wrong_top10_rate"] = wrong_top10_rate

        exposure_ranks = self.get_exposure_ranks(pred_list, topK_exposure=topK_exposure)

        for item_id, ranks in exposure_ranks.items():
            if len(ranks) == 0:
                continue
            exposure_count = len(ranks)
            avg_exposure_rank = float(np.mean(ranks))

            int_id = int(item_id)
            if int_id not in item_rank_stats:

                item_rank_stats[int_id] = {
                    "hit_count": 0,
                    "avg_pos_rank": 0.0,
                    "pos_top10_rate": 0.0,
                    "pos_top20_rate": 0.0,
                }

            item_rank_stats[int_id]["exposure_count_topK"] = exposure_count
            item_rank_stats[int_id]["avg_exposure_rank_topK"] = avg_exposure_rank

        return item_rank_stats


    def get_neg_items_ranks(self, batch_pred_lists, answers, topK=10):
        """
          item_id -> [rank1, rank2, ...]
        """
        batch_neg_ranks = defaultdict(list)
        num_users = len(batch_pred_lists)

        for i in range(num_users):
            pred_list = batch_pred_lists[i]
            true_set = set(answers[i])

            for rank_idx, item_id in enumerate(pred_list[:topK], start=1):
                if item_id not in true_set:
                    batch_neg_ranks[item_id].append(rank_idx)

        return batch_neg_ranks


    def get_exposure_ranks(self, batch_pred_lists, topK_exposure=50):

        exposure_ranks = defaultdict(list)

        for pred_list in batch_pred_lists:
            for rank_idx, item_id in enumerate(pred_list[:topK_exposure], start=1):
                exposure_ranks[item_id].append(rank_idx)

        return exposure_ranks



class FinetuneTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
        self._agm_t = 0
        self._agm_hat_s_img = 0.0
        self._agm_hat_s_txt = 0.0



    def _grad_l2_norm(self, named_params, keywords):

        sq_sum = 0.0
        found = False
        for name, p in named_params:
            if (p.grad is None) or (not p.requires_grad):
                continue
            n = name.lower()
            if any(k in n for k in keywords):
                g = p.grad.detach()
                sq_sum += float(torch.sum(g * g).item())
                found = True
        if not found:
            return float("nan")
        return math.sqrt(sq_sum)

    def _collect_modal_grad_norms(self):

        named_params = list(self.model.named_parameters())

        g_img = self._grad_l2_norm(named_params, ["image_", "img", "vision"])

        g_txt = self._grad_l2_norm(named_params, ["text_", "txt"])

        g_id  = self._grad_l2_norm(named_params, [
            "item_mean_embeddings", "item_cov_embeddings",
            "position_embeddings",  
            "item_encoder", "encoder"  
        ])
        return g_img, g_txt, g_id


    @torch.no_grad()
    def _agm_compute_kappas(self, mono_loss_img, mono_loss_txt):

        alpha = float(self.args.agm_alpha)

        s_img = -mono_loss_img.detach()
        s_txt = -mono_loss_txt.detach()

        r_img = torch.exp(s_img - s_txt)
        r_txt = torch.exp(s_txt - s_img)

        self._agm_t += 1
        t = self._agm_t
        self._agm_hat_s_img = self._agm_hat_s_img * (t - 1) / t + float(s_img) / t
        self._agm_hat_s_txt = self._agm_hat_s_txt * (t - 1) / t + float(s_txt) / t

        tau_img = np.exp(self._agm_hat_s_img - self._agm_hat_s_txt)
        tau_txt = np.exp(self._agm_hat_s_txt - self._agm_hat_s_img)

        tau_img = torch.tensor(tau_img, device=r_img.device, dtype=r_img.dtype)
        tau_txt = torch.tensor(tau_txt, device=r_txt.device, dtype=r_txt.dtype)

        kappa_img = torch.exp(-alpha * (r_img - tau_img))
        kappa_txt = torch.exp(-alpha * (r_txt - tau_txt))

        kappa_img = torch.clamp(kappa_img, self.args.agm_kappa_min, self.args.agm_kappa_max)
        kappa_txt = torch.clamp(kappa_txt, self.args.agm_kappa_min, self.args.agm_kappa_max)

        return kappa_img, kappa_txt

    def _agm_scale_grads(self, kappa_img, kappa_txt):

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            n = name.lower()

            is_img = ("image" in n) or ("task2" in n) or ("dnn2" in n)

            is_txt = ("text" in n) or ("task3" in n) or ("dnn3" in n)

            if is_img and (not is_txt):
                p.grad.mul_(kappa_img)
            elif is_txt and (not is_img):
                p.grad.mul_(kappa_txt)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        rec_data_iter = dataloader
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_auc = 0.0

            rec_avg_g_img = 0.0
            rec_avg_g_txt = 0.0
            rec_avg_g_id  = 0.0

            rec_avg_kappa_img = 0.0
            rec_avg_kappa_txt = 0.0
            rec_cnt_kappa = 0


            # for i, batch in rec_data_iter:
            for batch in tqdm(rec_data_iter):
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch


                # ========= 1)  mono losses -> kappas=========
                if self.args.use_agm and self.args.is_use_mm:
                    with torch.no_grad():
                        # Image-only (ID + Image -> backbone)
                        seq_img, _ = self.model.finetune(input_ids, modal="img")
                        loss_img, _ = self.cross_entropy(seq_img, target_pos, target_neg)

                        # Text-only (ID + Text -> backbone)
                        seq_txt, _ = self.model.finetune(input_ids, modal="txt")
                        loss_txt, _ = self.cross_entropy(seq_txt, target_pos, target_neg)

                    kappa_img, kappa_txt = self._agm_compute_kappas(loss_img, loss_txt)
                else:
                    kappa_img = kappa_txt = None
                
                if kappa_img is not None:
                    rec_avg_kappa_img += float(kappa_img.item())
                    rec_avg_kappa_txt += float(kappa_txt.item())
                    rec_cnt_kappa += 1

                # ========= 2) full loss =========
                # Binary cross_entropy
                # sequence_output, _ = self.model.finetune(input_ids)
                sequence_output, _ = self.model.finetune(input_ids, modal="full")
                loss, batch_auc = self.cross_entropy(sequence_output, target_pos, target_neg)
                
                self.optim.zero_grad()
                loss.backward()

                # ========= 3) AGM: scale image/text ）=========
                if kappa_img is not None:
                    self._agm_scale_grads(kappa_img, kappa_txt)

                g_img, g_txt, g_id = self._collect_modal_grad_norms()
                rec_avg_g_img += 0.0 if math.isnan(g_img) else g_img
                rec_avg_g_txt += 0.0 if math.isnan(g_txt) else g_txt
                rec_avg_g_id  += 0.0 if math.isnan(g_id)  else g_id

                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.4f}'.format(rec_avg_auc / len(rec_data_iter)),
            }
            post_fix.update({
                "image_grad_norm": '{:.6f}'.format(rec_avg_g_img / len(rec_data_iter)),
                "text_grad_norm":  '{:.6f}'.format(rec_avg_g_txt / len(rec_data_iter)),
                "id_grad_norm":    '{:.6f}'.format(rec_avg_g_id  / len(rec_data_iter)),
            })
            if rec_cnt_kappa > 0:
                post_fix.update({
                    "kappa_img": '{:.4f}'.format(rec_avg_kappa_img / rec_cnt_kappa),
                    "kappa_txt": '{:.4f}'.format(rec_avg_kappa_txt / rec_cnt_kappa),
                })

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                #  for i, batch in rec_data_iter:
                i = 0
                for batch in tqdm(rec_data_iter):
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output, _ = self.model.finetune(input_ids)

                    recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind = np.argpartition(rating_pred, -40)[:, -40:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                    i += 1
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                #  for i, batch in rec_data_iter:
                i = 0
                for batch in tqdm(rec_data_iter):
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    # print(input_ids)
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                    i += 1

                return self.get_sample_scores(epoch, pred_list)


class STOSATrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(STOSATrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
        self._agm_t = 0
        self._agm_hat_s_img = 0.0
        self._agm_hat_s_txt = 0.0

    def bpr_optimization(self, seq_mean_out, seq_cov_out, pos_ids, neg_ids):  
        # print("bpr_optimization")
        # [batch seq_len hidden_size]
        activation = nn.ELU()
        pos_mean_emb = self.model.item_mean_embeddings(pos_ids)
        pos_cov_emb = activation(self.model.item_cov_embeddings(pos_ids)) + 1
        neg_mean_emb = self.model.item_mean_embeddings(neg_ids)
        neg_cov_emb = activation(self.model.item_cov_embeddings(neg_ids)) + 1

        # [batch*seq_len hidden_size]
        pos_mean = pos_mean_emb.view(-1, pos_mean_emb.size(2))
        pos_cov = pos_cov_emb.view(-1, pos_cov_emb.size(2))
        neg_mean = neg_mean_emb.view(-1, neg_mean_emb.size(2))
        neg_cov = neg_cov_emb.view(-1, neg_cov_emb.size(2))
        seq_mean_emb = seq_mean_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        seq_cov_emb = seq_cov_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]

        if self.args.distance_metric == 'wasserstein':
            pos_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = wasserstein_distance(pos_mean, pos_cov, neg_mean, neg_cov)

        else:
            pos_logits = kl_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = kl_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = kl_distance(pos_mean, pos_cov, neg_mean, neg_cov)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(-torch.log(torch.sigmoid(neg_logits - pos_logits + 1e-24)) * istarget) / torch.sum(istarget)
        pvn_loss = self.args.pvn_weight * torch.sum(torch.clamp(pos_logits - pos_vs_neg, 0) * istarget) / torch.sum(istarget)
        auc = torch.sum(
            ((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc, pvn_loss

    def dist_predict_full(self, seq_mean_out, seq_cov_out):  
        # print("dist_predict_full")
        elu_activation = torch.nn.ELU()
        test_item_mean_emb = self.model.item_mean_embeddings.weight
        test_item_cov_emb = elu_activation(self.model.item_cov_embeddings.weight) + 1

        return wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)


    def _grad_l2_norm(self, named_params, keywords):

        sq_sum = 0.0
        found = False
        for name, p in named_params:
            if (p.grad is None) or (not p.requires_grad):
                continue
            n = name.lower()
            if any(k in n for k in keywords):
                g = p.grad.detach()
                sq_sum += float(torch.sum(g * g).item())
                found = True
        if not found:
            return float("nan")
        return math.sqrt(sq_sum)

    def _collect_modal_grad_norms(self):

        named_params = list(self.model.named_parameters())

        # 1) Image 
        g_img = self._grad_l2_norm(named_params, ["image_", "img", "vision"])

        # 2) Text 
        g_txt = self._grad_l2_norm(named_params, ["text_", "txt"])

        # 3) ID（item_embeddings / user_embeddings / position_embeddings / encoder）
        # g_id = self._grad_l2_norm(named_params, ["item_embeddings", "user_embeddings", "position_embeddings"])
        g_id  = self._grad_l2_norm(named_params, [
            "item_mean_embeddings", "item_cov_embeddings",
            "position_embeddings",  
            "item_encoder", "encoder"  
        ])
        return g_img, g_txt, g_id


    @torch.no_grad()
    def _agm_compute_kappas(self, mono_loss_img, mono_loss_txt):

        alpha = float(self.args.agm_alpha)

        s_img = -mono_loss_img.detach()
        s_txt = -mono_loss_txt.detach()

        r_img = torch.exp(s_img - s_txt)
        r_txt = torch.exp(s_txt - s_img)

        # running average hat_s
        self._agm_t += 1
        t = self._agm_t
        self._agm_hat_s_img = self._agm_hat_s_img * (t - 1) / t + float(s_img) / t
        self._agm_hat_s_txt = self._agm_hat_s_txt * (t - 1) / t + float(s_txt) / t

        tau_img = np.exp(self._agm_hat_s_img - self._agm_hat_s_txt)
        tau_txt = np.exp(self._agm_hat_s_txt - self._agm_hat_s_img)

        tau_img = torch.tensor(tau_img, device=r_img.device, dtype=r_img.dtype)
        tau_txt = torch.tensor(tau_txt, device=r_txt.device, dtype=r_txt.dtype)

        kappa_img = torch.exp(-alpha * (r_img - tau_img))
        kappa_txt = torch.exp(-alpha * (r_txt - tau_txt))

        kappa_img = torch.clamp(kappa_img, self.args.agm_kappa_min, self.args.agm_kappa_max)
        kappa_txt = torch.clamp(kappa_txt, self.args.agm_kappa_min, self.args.agm_kappa_max)

        return kappa_img, kappa_txt

    def _agm_scale_grads(self, kappa_img, kappa_txt):

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            n = name.lower()

            is_img = ("image" in n) or ("task2" in n) or ("dnn2" in n)

            is_txt = ("text" in n) or ("task3" in n) or ("dnn3" in n)

            if is_img and (not is_txt):
                p.grad.mul_(kappa_img)
            elif is_txt and (not is_img):
                p.grad.mul_(kappa_txt)



    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        rec_data_iter = dataloader

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_pvn_loss = 0.0
            rec_avg_auc = 0.0
            # rec_avg_bal_loss = 0.0
            rec_avg_g_img = 0.0
            rec_avg_g_txt = 0.0
            rec_avg_g_id  = 0.0



            # for batch in rec_data_iter:
            for batch in tqdm(rec_data_iter, desc="Training Progress"):
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, _ = batch
                # # (1) mono -> kappas (no grad)
                # if self.args.use_agm and self.args.is_use_mm:
                #     with torch.no_grad():
                #         seq_mean_img, seq_cov_img, _, _ = self.model.finetune(input_ids, user_ids, modal="img")
                #         loss_img, _, pvn_img = self.bpr_optimization(seq_mean_img, seq_cov_img, target_pos, target_neg)
                #         mono_img = loss_img + pvn_img

                #         seq_mean_txt, seq_cov_txt, _, _ = self.model.finetune(input_ids, user_ids, modal="txt")
                #         loss_txt, _, pvn_txt = self.bpr_optimization(seq_mean_txt, seq_cov_txt, target_pos, target_neg)
                #         mono_txt = loss_txt + pvn_txt

                #     kappa_img, kappa_txt = self._agm_compute_kappas(mono_img, mono_txt)
                # else:
                #     kappa_img = kappa_txt = None
                
                # bpr optimization
                # sequence_mean_output, sequence_cov_output, _, _ = self.model.finetune(input_ids, user_ids)
                sequence_mean_output, sequence_cov_output, _, _ = self.model.finetune(input_ids, user_ids, modal="full")
                loss, batch_auc, pvn_loss = self.bpr_optimization(sequence_mean_output, sequence_cov_output, target_pos, target_neg)

                loss = loss + pvn_loss
                self.optim.zero_grad()
                loss.backward()


                g_img, g_txt, g_id = self._collect_modal_grad_norms()
                rec_avg_g_img += 0.0 if math.isnan(g_img) else g_img
                rec_avg_g_txt += 0.0 if math.isnan(g_txt) else g_txt
                rec_avg_g_id  += 0.0 if math.isnan(g_id)  else g_id

                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item()
                rec_avg_pvn_loss += pvn_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.6f}'.format(rec_avg_auc / len(rec_data_iter)),
                "rec_avg_pvn_loss": '{:.6f}'.format(rec_avg_pvn_loss / len(rec_data_iter)),
                # "rec_avg_bal_loss": '{:.6f}'.format(rec_avg_bal_loss / len(rec_data_iter)),
                # "gate_img_mean": '{:.4f}'.format(rec_avg_g_img / len(rec_data_iter)),
                # "gate_txt_mean": '{:.4f}'.format(rec_avg_g_txt / len(rec_data_iter)),
            }
            post_fix.update({
                "image_grad_norm": '{:.6f}'.format(rec_avg_g_img / len(rec_data_iter)),
                "text_grad_norm":  '{:.6f}'.format(rec_avg_g_txt / len(rec_data_iter)),
                "id_grad_norm":    '{:.6f}'.format(rec_avg_g_id  / len(rec_data_iter)),
            })


            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')
        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                with torch.no_grad():
                    # for i, batch in rec_data_iter:
                    i = 0
                    for batch in tqdm(rec_data_iter):
                        # 0. batch_data will be sent into the device(GPU or cpu)
                        batch = tuple(t.to(self.device) for t in batch)
                        user_ids, input_ids, target_pos, target_neg, answers = batch
                        recommend_mean_output, recommend_cov_output, _, _, = self.model.finetune(input_ids, user_ids)

                        recommend_mean_output = recommend_mean_output[:, -1, :]
                        recommend_cov_output = recommend_cov_output[:, -1, :]

                        rating_pred = self.dist_predict_full(recommend_mean_output, recommend_cov_output)
                        rating_pred = rating_pred.cpu().data.numpy().copy()
                        batch_user_index = user_ids.cpu().numpy()
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 1e+24
                        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                        ind = np.argpartition(rating_pred, 40)[:, :40]
                        # ind = np.argpartition(rating_pred, 100)[:, :100]
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        # ascending order
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::]
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                        if i == 0:
                            pred_list = batch_pred_list
                            answer_list = answers.cpu().data.numpy()
                        else:
                            pred_list = np.append(pred_list, batch_pred_list, axis=0)
                            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                        i += 1
                    return self.get_full_sort_score(epoch, answer_list, pred_list)




class HM4SRTrainer(Trainer):
    def iteration(self, epoch, dataloader, full_sort=True, train=True):
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            for batch in tqdm(dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                # train: (user_id, input_ids, target_pos, target_neg, answer, timestamp_list)
                user_ids, input_ids, target_pos, target_neg, answers, timestamp_list = batch

                # ===== DEBUG: range check BEFORE model call =====
                ids = input_ids.detach().cpu()
                tp  = target_pos.detach().cpu()
                ts  = timestamp_list.detach().cpu()

                # print("[DEBUG] ids:", ids.min().item(), ids.max().item(), "shape", tuple(ids.shape))
                # print("[DEBUG] tp :", tp.min().item(), tp.max().item(), "shape", tuple(tp.shape))
                # print("[DEBUG] ts :", ts.min().item(), ts.max().item(), "shape", tuple(ts.shape))

                assert ids.min().item() >= 0, "input_ids has negative"
                assert ids.max().item() < self.args.item_size, f"input_ids max {ids.max().item()} >= item_size {self.args.item_size}"

                assert tp.min().item() >= 0, "target_pos has negative"
                assert tp.max().item() < self.args.item_size, f"target_pos max {tp.max().item()} >= item_size {self.args.item_size}"

                assert ts.min().item() >= 0, "timestamp_list has negative"

                loss = self.model.calculate_loss(input_ids, target_pos, timestamp_list)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()

            post_fix = {"epoch": epoch, "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(dataloader))}
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)
            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')
            return post_fix

        else:
            self.model.eval()
            pred_list = None
            answer_list = None
            i = 0
            for batch in tqdm(dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                if len(batch) == 6:
                    user_ids, input_ids, target_pos, target_neg, answers, timestamp_list = batch
                    scores = self.model.full_sort_predict(input_ids, timestamp_list)
                else:
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs, timestamp_list = batch
                    scores = self.model.full_sort_predict(input_ids, timestamp_list)

                scores = scores.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                scores[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                ind = np.argpartition(scores, -40)[:, -40:]
                arr_ind = scores[np.arange(len(scores))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(scores)), ::-1]
                batch_pred_list = ind[np.arange(len(scores))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                i += 1

            return self.get_full_sort_score(epoch, answer_list, pred_list)

