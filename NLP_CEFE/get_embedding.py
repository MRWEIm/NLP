import argparse
import torch
from tqdm import tqdm
import utils
from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model/Qwen3-Embedding-4B')
    parser.add_argument('--train_data_path', type=str, default=f'./NLP_CEFE/data/train.json')
    parser.add_argument('--valid_data_path', type=str, default=f'./NLP_CEFE/data/val.json')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--conv_out_channels', type=int, default=50)
    parser.add_argument('--kernal_size', type=int, default=5)
    parser.add_argument('--lstm_hid_dim', type=int, default=50)
    parser.add_argument('--att_hid_dim', type=int, default=50)
    parser.add_argument('--dense_hid_dim', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--pos_weight', type=float, default=3.0)
    args = parser.parse_args()

    train_data = utils.load_data(args.train_data_path)
    val_data = utils.load_data(args.valid_data_path)

    train_data_processed = utils.process_data(train_data)
    val_data_processed = utils.process_data(val_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModel.from_pretrained(args.model_path).to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    train_tensor = []
    for s in tqdm(train_data_processed['sent_list']):
        inputs = tokenizer(s, return_tensors="pt", truncation=True, padding=False, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [1, hidden]
        train_tensor.append(cls_embedding)
    train_tensor = torch.cat(train_tensor, dim=0)
    torch.save(train_tensor, './NLP_CEFE/tensor/train_tensor.pt')

    val_tensor = []
    for s in tqdm(val_data_processed['sent_list']):
        inputs = tokenizer(s, return_tensors="pt", truncation=True, padding=False, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [1, hidden]
        val_tensor.append(cls_embedding)
    val_tensor = torch.cat(val_tensor, dim=0)
    torch.save(val_tensor, './NLP_CEFE/tensor/val_tensor.pt')
    print(train_tensor.shape, val_tensor.shape)


if __name__ == '__main__':
    main()