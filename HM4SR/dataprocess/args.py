import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    ### 基本参数
    parser.add_argument('--dataset', type=str, default='Games')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--random_seed', type=int, default=2024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tolerance', type=int, default=10)
    ### 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--seq_head', type=int, default=2)
    parser.add_argument('--seq_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    ### 路径参数
    args.txt_emb = f'./dataset/{args.dataset}/txt_emb.pt'
    args.img_emb = f'./dataset/{args.dataset}/img_emb.pt'
    args.ckpt = f'./ckpt/{args.dataset}'
    args.item_dict_path = f'./dataset/{args.dataset}/item2id'
    args.log_path = f'./log/{args.dataset}_{args.random_seed}.txt'
    args.inter_path = f'./dataset/{args.dataset}/{args.dataset}.inter'
    args.data_path = f'./dataset/{args.dataset}/00_seq'
    args.txt_path = f'./dataset/{args.dataset}/content.txt'
    args.img_path = f'./dataset/{args.dataset}/image/'
    args.meta_path = f'./dataset/meta_{args.dataset}.json.gz'
    return args