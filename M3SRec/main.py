# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import SASRecDataset, UserLLMDataset
from trainers import FinetuneTrainer, STOSATrainer
from M3SRec.M3SRec import SASRecModel, STOSA
from utils import EarlyStopping, get_user_seqs, check_path, set_seed
import time
import os
from datetime import datetime
from MMMLP import MMMLPModel

base_dir = os.path.dirname(os.path.abspath(__file__))  
data_name = 'Home'  # 'Beauty' 'Sports' 'Home'
prompt = "PromptV2"  # "Features" "PromptV1" "PromptV2"


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=f'./data/Features/{data_name}/', type=str)
    parser.add_argument('--output_dir', default=f'outputs/{data_name}', type=str)
    parser.add_argument('--data_name', default=f'reviews_{data_name}', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")
    parser.add_argument('--patience', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='SASRec', type=str)#Finetune_full
    parser.add_argument("--hidden_size", type=int, default=256, help="hidden size of transformer model") #64
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of layers") # default=1
    parser.add_argument('--num_attention_heads', default=4, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.0, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=100, type=int)
    parser.add_argument('--distance_metric', default='wasserstein', type=str)
    parser.add_argument('--pvn_weight', default=0.005, type=float)
    parser.add_argument('--kernel_param', default=1.0, type=float)

    # loss balance args
    parser.add_argument('--lambda_bal', default=0.01, type=float)
    parser.add_argument('--bal_margin', default=0.1, type=float)
    parser.add_argument('--use_agm', default=True, type=bool)
    parser.add_argument('--agm_alpha', type=float, default=1.0)
    parser.add_argument('--agm_kappa_min', type=float, default=0.2, help='Minimum clamp value for AGM kappa')
    parser.add_argument('--agm_kappa_max', type=float, default=2.0, help='Maximum clamp value for AGM kappa')


    # multimodal args
    parser.add_argument('--image_emb_path', default=f'data/Features/{data_name}/qwen25vl_promptV2_image_features.pt', type=str)
    # parser.add_argument('--image_emb_path', default=f'data/Features/{data_name}/clip_image_features.pt', type=str)
    parser.add_argument('--text_emb_path', default=f'data/Features/{data_name}/clip_text_features.pt', type=str)
    parser.add_argument('--mm_emb_dim', default=512, type=int)
    parser.add_argument("--is_use_mm", type=bool, default=True, help="is use mm embedding")
    parser.add_argument("--is_use_text", type=bool, default=False, help="is use text embedding")
    parser.add_argument("--is_use_image", type=bool, default=False, help="is use image embedding")
    parser.add_argument("--pretrain_emb_dim", type=int, default=512, help="pretrain_emb_dim of clip model")
    parser.add_argument("--pretrain_img_emb_dim", type=int, default=3584, help="pretrain_emb_dim of clip model") # 3584
    parser.add_argument("--is_use_cross", type=bool, default=True, help="is use mm cross")
    parser.add_argument('--num_shared_experts', default=2, type=int, help="shared experts for multi-modal fusion")
    parser.add_argument('--num_specific_experts', default=4, type=int, help="specific experts for multi-modal fusion")
    parser.add_argument('--low_rank', default=4, type=int, help="low_rank matrix")
    parser.add_argument('--global_transformer_nhead', default=4, type=int)
    parser.add_argument("--prediction", type=bool, default=False, help="activate prediction mode")

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="6", help="gpu_id")


    parser.add_argument("--hm4sr_n_layers", type=int, default=2)
    parser.add_argument("--hm4sr_n_heads", type=int, default=2)
    parser.add_argument("--hm4sr_inner_size", type=int, default=256)

    parser.add_argument("--hm4sr_temperature", type=float, default=0.2)
    parser.add_argument("--hm4sr_phcl_temperature", type=float, default=0.2)
    parser.add_argument("--hm4sr_phcl_weight", type=float, default=1.0)
    parser.add_argument("--hm4sr_beta", type=float, default=0.1)

    # MoE args
    parser.add_argument("--hm4sr_start_expert_num", type=int, default=3)
    parser.add_argument("--hm4sr_start_gate_selection", type=str, default="softmax")
    parser.add_argument("--hm4sr_initializer_weight", type=float, nargs=3, default=[1.0, 1.0, 1.0])

    parser.add_argument("--hm4sr_temporal_expert_num", type=int, default=4)
    parser.add_argument("--hm4sr_temporal_gate_selection", type=str, default="softmax")
    parser.add_argument("--hm4sr_interval_scale", type=float, default=1.0)

    # data paths（cat/time/stat）
    parser.add_argument("--hm4sr_cat_path", type=str, default=f"data/Features/{data_name}/cat.pt")
    parser.add_argument("--time_file", type=str, default=f"./data/Features/{data_name}/{data_name}_time.txt")
    parser.add_argument("--hm4sr_stat_dir", type=str, default=f"data/Features/{data_name}")




    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
        get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.num_users = num_users
    args.mask_id = max_item + 1
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    # save model args
    # prompt_name = Path(args.image_emb_path).stem # 'clip_image_features_prompt_Sports'
    prompt_name = prompt
    
    args.clean_data_name = args.data_name.replace('reviews_', '')
    
    args_str = f'{args.model_name}-{args.clean_data_name}-{args.hidden_size}-{prompt_name}' 
    args.log_file = os.path.join(args.output_dir, f'{args_str}_{timestamp}.txt') # STOSA-Beauty-userprompt_20251202.txt

    # print(str(args))
    # with open(args.log_file, 'a') as f:
    #     f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = f'{args_str}_{prompt_name}_{timestamp}.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)


    # if args.model_name == "HM4SR":
    #     from utils import get_user_time_seqs
    #     from datasets import HM4SRDataset

    #     user_time_seq = get_user_time_seqs(args.time_file)
        
    #     train_dataset = HM4SRDataset(args, user_seq, user_time_seq, data_type="train")
    #     train_sampler = RandomSampler(train_dataset)
    #     train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    #     eval_dataset  = HM4SRDataset(args, user_seq, user_time_seq, data_type="valid")
    #     eval_sampler = SequentialSampler(eval_dataset)

    #     test_dataset  = HM4SRDataset(args, user_seq, user_time_seq, data_type="test")
    #     test_sampler = SequentialSampler(test_dataset)
    # else:
    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)

    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)

    
    # ==================================
    if args.model_name == 'STOSA':
        model = STOSA(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=100)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)
        trainer = STOSATrainer(model, train_dataloader, eval_dataloader,
                                    test_dataloader, args)

    elif args.model_name == "HM4SR":
        from M3SRec.M3SRec import HM4SR
        from trainers import HM4SRTrainer
        model = HM4SR(args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=100)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)
        trainer = HM4SRTrainer(model, train_dataloader, eval_dataloader, 
                                    test_dataloader, args)

    elif args.model_name == "MMMLP":
        model = MMMLPModel(args=args)

        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

        trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, args)

    else:
        model = SASRecModel(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

        trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, args)

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info, _, _ = trainer.test(0, full_sort=True)

    else:
        if args.model_name == 'STOSA':
            early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        else:
            early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):

            trainer.train(epoch)
            start_time_epoch_valid = time.time()
            scores,  _, _ , _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)

            end_time_epoch_valid = time.time()
            epoch_duration = end_time_epoch_valid - start_time_epoch_valid
            x_epoch_duration = format(epoch_duration, ".2f")
            print(f"Epoch {epoch} valid in {epoch_duration:.2f} seconds.")

            with open(args.log_file, 'a') as f:
                f.write(f"Epoch {epoch} duration: {epoch_duration:.2f} seconds\n")
                f.write(x_epoch_duration + '\n')

            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Change to test_rating_matrix!-------------------')

        # Load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        valid_scores, _, _, _ = trainer.valid('best', full_sort=True)
        trainer.args.train_matrix = test_rating_matrix

        # Start timing the testing phase
        start_time_test = time.time()
        scores, result_info, _, item_rank_stats = trainer.test('best', full_sort=True)
        end_time_test = time.time()


        print(f"[DEBUG] num_items_with_any_feedback = {len(item_rank_stats)}")
        num_pos = sum(1 for v in item_rank_stats.values() if v.get("hit_count", 0) > 0)
        num_neg = sum(1 for v in item_rank_stats.values() if v.get("wrong_hit_topK", 0) > 0)
        num_expo = sum(1 for v in item_rank_stats.values() if v.get("exposure_count_topK", 0) > 0)
        print(f"[DEBUG] items_with_positive_feedback = {num_pos}")
        print(f"[DEBUG] items_with_negative_topK_feedback = {num_neg}")
        print(f"[DEBUG] items_with_exposure_feedback = {num_expo}")


        rank_stats_path = os.path.join(
            f'./data/{data_name}/',
            f"{data_name}_{prompt}_item_rank.json"
        )
        with open(rank_stats_path, "w") as f:
            json.dump(item_rank_stats, f, indent=2)
        print(f"[INFO] Saved item rank feedback to {rank_stats_path}")
        # # ====================================================

        # Calculate and log the prediction time
        prediction_duration = end_time_test - start_time_test
        print(f"Prediction time: {prediction_duration:.2f} seconds.")
        with open(args.log_file, 'a') as f:
            f.write(f"Prediction time: {prediction_duration:.2f} seconds\n")

    # Log total training time
    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
        f.write(f"Total training time: {int(minutes):02d}:{int(seconds):02d}" + '\n')

    print(f"Total training time: {int(minutes):02d}:{int(seconds):02d}")


main()
