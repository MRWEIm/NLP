import argparse

from matplotlib import pyplot as plt
import utils
import torch
import itertools
from tqdm import tqdm
from copy import deepcopy
from loss_fn import loss_func, AutomaticWeightedLoss
from CEFE_model import CEFE
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class CEFE_dataset(Dataset):
    def __init__(self, data, output_type):
        self.sent_list = data['sent_list']
        self.sent_pos_tensor = data['sent_pos_tensor']
        self.courseGrainedErrorLabel = data['courseGrainedError_label_tensor']
        self.fineGrainedErrorLabel = data['fineGrainedError_label_tensor']

        self.output_type = output_type
        assert output_type in ["two_level", "single_level"]
    def __len__(self):
        return len(self.sent_list)
    
    def __getitem__(self, index):
        sent = self.sent_list[index]
        return sent, self.sent_pos_tensor[index], self.courseGrainedErrorLabel[index], self.fineGrainedErrorLabel[index]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model/chinese-bert-wwm')
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
    parser.add_argument('--pos_weight', type=float, default=2.0)
    args = parser.parse_args()

    train_data = utils.load_data(args.train_data_path)
    val_data = utils.load_data(args.valid_data_path)
    print('Getting train data')
    train_data_processed = utils.process_data(train_data)
    train_dataset = CEFE_dataset(train_data_processed, output_type='two_level')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print('Getting val data')
    val_data_processed = utils.process_data(val_data)
    val_dataset = CEFE_dataset(val_data_processed, output_type='two_level')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    param_grid = {
        'epoch': [3, 5, 10],
        'pos_weight': [3.0, 4.0, 5.0, 6.0, 7.0]
    }

    best_combination = {'f1': 0}
    f1_scores = []
    keys, values = zip(*param_grid.items())
    for combo in tqdm(itertools.product(*values), total=len(list(itertools.product(*values))), desc="Grid Search"):
        combo_dict = dict(zip(keys, combo))

        args.epoch = combo_dict['epoch']
        args.pos_weight = combo_dict['pos_weight']

        f1_list = []
        for _ in range(10):
            model = CEFE(args=args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            adaptive_loss = AutomaticWeightedLoss(num=2)

            for _ in range(args.epoch):
                model.train()
                for batch in train_dataloader:
                    sent, sent_pos_tensor, courseLabel, fineLabel = batch
                    sent_encoded_inputs = tokenizer(sent, padding='max_length', max_length=256, truncation=True, return_tensors='pt')
                    sent_encoded_inputs, sent_pos_tensor, courseLabel, fineLabel = map(lambda x: x.to(device), (sent_encoded_inputs, sent_pos_tensor, courseLabel, fineLabel))
                    coursePred, finePred = model(sent_encoded_inputs, sent_pos_tensor)
                    loss = loss_func(coursePred, courseLabel, finePred, fineLabel, args.pos_weight)
                    loss = adaptive_loss(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
            model.eval()
            coursePredList, finePredList, courseLabelList, fineLabelList = [], [], [], []
            with torch.no_grad():
                for batch in val_dataloader:
                    sent, sent_pos_tensor, courseLabel, fineLabel = batch
                    sent_encoded_inputs = tokenizer(sent, padding='max_length', max_length=256, truncation=True, return_tensors='pt')
                    sent_encoded_inputs, sent_pos_tensor = map(lambda x: x.to(device), (sent_encoded_inputs, sent_pos_tensor))
                    coursePred, finePred = model(sent_encoded_inputs, sent_pos_tensor)
                    coursePredList.append(coursePred)
                    finePredList.append(finePred)
                    courseLabelList.append(courseLabel)
                    fineLabelList.append(fineLabel)
                coursePredList = torch.cat(coursePredList, dim=0)
                finePredList = torch.cat(finePredList, dim=0)
                courseLabelList = torch.cat(courseLabelList, dim=0)
                fineLabelList = torch.cat(fineLabelList, dim=0)
                f1, _, _ = utils.evaluate(coursePredList, courseLabelList, finePredList, fineLabelList)
                f1_list.append(f1)

        avg_f1 = sum(f1_list) / len(f1_list)
        f1_scores.append(avg_f1)

        if avg_f1 > best_combination['f1']:
            best_combination = {'f1': f1, 'params': deepcopy(combo_dict)}

    print(f"Best config: {best_combination['params']}, Best F1: {best_combination['f1']:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(param_grid['pos_weight'], f1_scores, marker='o', linestyle='-', color='b')
    plt.title('F1 Score vs pos_weight')
    plt.xlabel('pos_weight')
    plt.ylabel('F1 Score')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('f1_vs_pos_weight.png')
    plt.show()
    return

if __name__ == '__main__':
    main()