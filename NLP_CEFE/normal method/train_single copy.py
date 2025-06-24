import argparse

from matplotlib import pyplot as plt
import utils
import torch
import itertools
from tqdm import tqdm
from copy import deepcopy
from loss_fn import loss_func
from CEFE_model import CEFE, CEFE_pos, CEFE_bert
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
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--pos_weight', type=float, default=3.0)
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

    model_list = [CEFE(args=args).to(device), CEFE_pos(args=args).to(device), CEFE_bert(args=args).to(device)]

    history = {
        0: {"f1": [], "course_f1": [], "fine_f1": []},
        1: {"f1": [], "course_f1": [], "fine_f1": []},
        2: {"f1": [], "course_f1": [], "fine_f1": []}
    }

    for i in range(3):
        model = model_list[i]
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        f1_history = []
        course_f1_history = []
        fine_f1_history = []

        for _ in range(args.epoch):
            model.train()
            for batch in train_dataloader:
                sent, sent_pos_tensor, courseLabel, fineLabel = batch
                sent_encoded_inputs = tokenizer(sent, padding='max_length', max_length=256, truncation=True, return_tensors='pt')
                sent_encoded_inputs, sent_pos_tensor, courseLabel, fineLabel = map(lambda x: x.to(device), (sent_encoded_inputs, sent_pos_tensor, courseLabel, fineLabel))
                coursePred, finePred = model(sent_encoded_inputs, sent_pos_tensor)
                loss = loss_func(coursePred, courseLabel, finePred, fineLabel, args.pos_weight)
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
                f1, course_f1, fine_f1 = utils.evaluate(coursePredList, courseLabelList, finePredList, fineLabelList)

                print(f'f1: {f1:.4f} course_f1: {course_f1:.4f} fine_f1: {fine_f1:.4f}')

                f1_history.append(f1)
                course_f1_history.append(course_f1)
                fine_f1_history.append(fine_f1)
            
        # 将当前模型的指标保存到 history 中
        history[i]["f1"] = f1_history
        history[i]["course_f1"] = course_f1_history
        history[i]["fine_f1"] = fine_f1_history

    epochs = range(1, args.epoch + 1)

    # 创建一个包含 3 个子图的大图（纵向排列）
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 每个子图对应一个指标
    metrics = ['f1', 'course_f1', 'fine_f1']
    titles = ['F1 Score Over Epochs', 'Course F1 Score Over Epochs', 'Fine F1 Score Over Epochs']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for ax, metric, title, color in zip(axes, metrics, titles, itertools.cycle([colors])):
        for i in range(3):
            values = history[i][metric]
            ax.plot(epochs, values, label=f'Model {i}', color=colors[i], marker='o' if i == 0 else ('s' if i == 1 else '^'))
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(False)

    plt.savefig('all_metrics_comparison.png')  # 保存图像
    plt.show()

    return

if __name__ == '__main__':
    main()