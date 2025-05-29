import argparse
import utils
import torch
from loss_fn import loss_func
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
    parser.add_argument('--conv_out_channels', type=int, default=50)
    parser.add_argument('--lstm_hid_dim', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=20)
    args = parser.parse_args()

    train_data = utils.load_data(args.train_data_path)
    val_data = utils.load_data(args.valid_data_path)
    print('Getting train data')
    train_data_processed = utils.process_data(train_data)
    train_dataset = CEFE_dataset(train_data_processed, output_type='two_level')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    print('Getting val data')
    val_data_processed = utils.process_data(val_data)
    val_dataset = CEFE_dataset(val_data_processed, output_type='two_level')
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CEFE(args=args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print('='*50)
    for ii in range(args.epoch):
        model.train()
        for batch in train_dataloader:
            sent, sent_pos_tensor, courseLabel, fineLabel = batch
            sent_encoded_inputs = tokenizer(sent, padding='max_length', max_length=256, truncation=True, return_tensors='pt')
            sent_encoded_inputs, sent_pos_tensor, courseLabel, fineLabel = map(lambda x: x.to(device), (sent_encoded_inputs, sent_pos_tensor, courseLabel, fineLabel))
            coursePred, finePred = model(sent_encoded_inputs, sent_pos_tensor)
            loss = loss_func(coursePred, courseLabel, finePred, fineLabel)
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
            f1, f1_course, f1_fine = utils.evaluate(coursePredList, courseLabelList, finePredList, fineLabelList)
            print(f'Epoch: {ii+1}, total f1: {f1:.4f} course f1: {f1_course:.4f}, fine f1: {f1_fine:.4f}')
            print('='*50)

    return

if __name__ == '__main__':
    main()