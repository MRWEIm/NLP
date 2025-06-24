import json
import torch
import jieba
import jieba.posseg as pseg
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoTokenizer
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file) 
    return data
    
def process_data(data, LLM=False):
    courseGrainedErrorTypeList = ["字符级错误", "成分残缺型错误", "成分赘余型错误", "成分搭配不当型错误"]
    fineGrainedErrorTypeList = ["缺字漏字", "错别字错误", "缺少标点", "错用标点",
                                "主语不明", "谓语残缺", "宾语残缺", "其他成分残缺",
                                "主语多余", "虚词多余", "其他成分多余", "空缺",
                                "语序不当", "动宾搭配不当", "其他搭配不当", "空缺"]
    flag_dict = {
        'pad': 0, 'unk': 1, 'eng': 2, 'mq': 3, 'j': 4, 'nrfg': 5, 'vg': 6, 'nrt': 7, 'z': 8, 'nr': 9, 'ug': 10,
        'uj': 11, 'o': 12, 'n': 13, 'v': 14, 'm': 15, 'c': 16, 'p': 17, 'f': 18, 'a': 19, 'nt': 20,
        'i': 21, 'd': 22, 'k': 23, 'vn': 24, 'ns': 25, 'uv': 26, 'q': 27, 'b': 28, 'x': 29, 'y': 30,
        'df': 31, 'r': 32, 'ul': 33, 'ad': 34, 'an': 35, 'u': 36, 'ng': 37, 'zg': 38, 'nz': 39,
        'ud': 40, 'l': 41, 't': 42, 'uz': 43, 's': 44
    }

    courseGrainedError_label_tensor = []
    fineGrainedError_label_tensor = []
    sent_pos_tensor_list = []
    sent_list = []
    max_len = 0
    for index, d in enumerate(data):
        if isinstance(d, dict):
            sent = d['sent']
            if sent == 'error' or 'CourseGrainedErrorType' not in d.keys() or 'FineGrainedErrorType' not in d.keys():
                continue
            courseGrainedErrorType = d['CourseGrainedErrorType']
            fineGrainedErrorType = d['FineGrainedErrorType']

            if LLM == True:
                sent_list.append('病句：“因而，保存良好的家风，摒弃有害而无益的家风，是有助于人成长的一大益事。”，它存在成分搭配不当型错误：动宾搭配不当。\n \
                                病句：“狗狗事件不值一提了，但我奶奶您辛苦了，我以念念不忘的这件事来以此激励自己坚持走下去不让您老人家失望。”，它存在字符级错误：缺少标点；成分赘余型错误：主语多余、其他成分多余。\n \
                                下面这个句子存在语病，请告诉我它存在的语病类型:' + sent)
            else:
                sent_list.append(sent)

            sent_pos_tensor = []
            count = 0
            words = pseg.cut(sent)
            for word, flag in words:
                count += 1
                if flag in flag_dict:
                    sent_pos_tensor.append(flag_dict[flag])
                else:
                    sent_pos_tensor.append(flag_dict['unk'])
            max_len = max(max_len, count)
            sent_pos_tensor_list.append(sent_pos_tensor)

            courseGrainedError_label = [0] * 4
            for cge in courseGrainedErrorType:
                if cge in courseGrainedErrorTypeList:
                    index = courseGrainedErrorTypeList.index(cge)
                    courseGrainedError_label[index] = 1
            courseGrainedError_label_tensor.append(courseGrainedError_label)

            fineGrainedError_label = [0] * 16
            for fge in fineGrainedErrorType:
                if fge in fineGrainedErrorTypeList:
                    index = fineGrainedErrorTypeList.index(fge)
                    fineGrainedError_label[index] = 1
            fineGrainedError_label_tensor.append(fineGrainedError_label)
        
    courseGrainedError_label_tensor = torch.tensor(courseGrainedError_label_tensor, dtype=torch.float32)
    fineGrainedError_label_tensor = torch.tensor(fineGrainedError_label_tensor, dtype=torch.float32)
    print(f'sentence max length: {max_len}')

    sent_pos_tensor_list = pad_sequence(sent_pos_tensor_list, max(max_len, 70))
    sent_pos_tensor = torch.tensor(sent_pos_tensor_list)

    # print(f'course stat: {courseGrainedError_label_tensor.sum(dim=0)} \nfine stat: {fineGrainedError_label_tensor.sum(dim=0)}')
    processed_data = {'sent_list': sent_list, 'sent_pos_tensor': sent_pos_tensor, 
                     'courseGrainedError_label_tensor': courseGrainedError_label_tensor, 
                     'fineGrainedError_label_tensor': fineGrainedError_label_tensor}
    return processed_data

def pad_sequence(sent_pos_tensor_list, max_len):
    for i in range(len(sent_pos_tensor_list)):
        if len(sent_pos_tensor_list[i]) < max_len:
            sent_pos_tensor_list[i].extend([0] * (max_len - len(sent_pos_tensor_list[i])))
    return sent_pos_tensor_list


def evaluate(course_pred, course_true, fine_pred, fine_true):
    course_true = course_true.cpu().numpy()
    course_pred = course_pred.detach().cpu().numpy()
    course_pred = course_pred > 0.5
    course_pred = course_pred.astype(int)

    fine_pred = torch.cat([fine_pred[:, 0:11], fine_pred[:, 12:-1]], dim=1)
    fine_true = torch.cat([fine_true[:, 0:11], fine_true[:, 12:-1]], dim=1)

    fine_true = fine_true.cpu().numpy()
    fine_pred = fine_pred.detach().cpu().numpy()
    fine_pred = fine_pred > 0.5
    fine_pred = fine_pred.astype(int)

    course_f1_score = f1_score(course_true, course_pred, average='micro')
    fine_f1_score = f1_score(fine_true, fine_pred, average='micro')
    return 0.5 * course_f1_score + 0.5 * fine_f1_score, course_f1_score, fine_f1_score


if __name__ == '__main__':
    data = load_data('./NLP_CEFE/data/train.json')
    process_data(data)
