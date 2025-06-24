import argparse
import os
import numpy as np
from tqdm import tqdm
import utils
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from baukit import TraceDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

def get_llama_activations_bau(model, prompt, device): 
    model.eval()

    HEADS = [f"model.layers.{i}.self_attn.o_proj" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS+MLPS, retain_input=True) as ret:
            output = model(prompt, output_hidden_states = True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].input.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states


def tokenize_prompt(prompts, tokenizer):
    tokenized_prompts = []
    for p in prompts:
        p = tokenizer.encode(p, return_tensors='pt')
        tokenized_prompts.append(p)
    return tokenized_prompts


def get_activations(model, prompts, device):
    all_layer_wise_activations = []
    all_head_wise_activations = []
    for p in tqdm(prompts):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, p, device)
        all_layer_wise_activations.append(layer_wise_activations[1:, -1, :])
        all_head_wise_activations.append(head_wise_activations[:, -1, :])
    layer_wise_activations = torch.tensor(np.array(all_layer_wise_activations))
    head_wise_activations = torch.tensor(np.array(all_head_wise_activations))
    head_wise_activations = head_wise_activations.reshape(head_wise_activations.shape[0], head_wise_activations.shape[1], 32, 128)
    return layer_wise_activations, head_wise_activations


def draw_heapmap(data):
    error_type = ["字符级错误", "成分残缺型错误", "成分赘余型错误", "成分搭配不当型错误",
                  "缺字漏字", "错别字错误", "缺少标点", "错用标点",
                  "主语不明", "谓语残缺", "宾语残缺", "其他成分残缺",
                  "主语多余", "虚词多余", "其他成分多余",
                  "语序不当", "动宾搭配不当", "其他搭配不当"]
    save_dir = './NLP_CEFE/plot'
    num_images = data.shape[0]
    for i in range(num_images):
        sns.heatmap(data[i].cpu().numpy(), cmap='Blues')  # 可选 cmap: "viridis", "coolwarm", "Blues", "YlGnBu" 等
        plt.title("Heatmap Example")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        
        save_path = os.path.join(save_dir, f"heatmap_{error_type[i]}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存为高分辨率图片
        plt.close()  # 关闭图像，避免内存泄漏


def train_probe(data):
    head_activations = data['head_wise_act']
    course_label = data['courseGrainedError_label_tensor']
    fine_label = data['fineGrainedError_label_tensor']
    label = torch.cat((course_label, fine_label), dim=1) # [104, 20]

    all_pred_rst = []
    all_probe = []
    for s in tqdm(range(label.shape[-1]), unit='error'):
        if s not in [15, 19]:
            s_probe = []
            layer_pred_rst = []
            for l in range(32):
                head_pred_rst = []
                for h in range(32):
                    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
                    clf.fit(head_activations[:, l, h, :], label[:, s])
                    s_probe.append(clf)
                    train_pred = clf.predict(head_activations[:, l, h, :])
                    train_acc = f1_score(label[:, s].cpu().numpy(), train_pred)
                    head_pred_rst.append(train_acc)
                layer_pred_rst.append(head_pred_rst)
            all_pred_rst.append(layer_pred_rst) 
            all_probe.append(s_probe)
    
    all_pred_rst = torch.tensor(all_pred_rst)
    draw_heapmap(all_pred_rst)

    return all_pred_rst, all_probe

def get_top_heads(all_pred_rst, all_probe, k=32):
    all_pred_rst_flatten = all_pred_rst.reshape(18, -1)
    sorted_pred_rst, indices = torch.sort(all_pred_rst_flatten, dim=1, descending=True)
    indices = indices[:, :k].numpy() # [18, 32]
    
    top_k_clf = []
    for i in range(18):
        s_top_k_clf = []
        for j in range(k):
            s_top_k_clf.append(all_probe[i][indices[i, j]])
        top_k_clf.append(s_top_k_clf)

    top_k_heads = []
    for i in range(18):
        s_top_k_heads = []
        for j in range(k):
            layer = indices[i, j] // 32
            head = indices[i, j] % 32
            s_top_k_heads.append((layer, head))
        top_k_heads.append(s_top_k_heads)

    return top_k_heads, top_k_clf


def f1_cal(data, top_k_heads, top_k_clf, threshold=0.5, k=32):
    head_activations = data['head_wise_act']
    
    all_pred_rst = [] # [18, 32, val_data_size]
    for i in range(18):
        pred_rst = []
        for j in range(k):
            pred = top_k_clf[i][j].predict(head_activations[:, top_k_heads[i][j][0], top_k_heads[i][j][1], :])
            pred_rst.append(pred)
        all_pred_rst.append(pred_rst)

    all_pred_rst = np.array(all_pred_rst)
    counts = np.sum(all_pred_rst == 1, axis=1)
    result = (counts > threshold * all_pred_rst.shape[1]).astype(int).T
    result = np.insert(result, 15, 0, axis=1)
    zeros_column = np.zeros((27, 1))
    result = np.hstack([result, zeros_column])

    course_result, fine_result = result[:, :4], result[:, 4:]
    
    return course_result, fine_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model/Llama-2-7b-chat-hf')
    parser.add_argument('--train_data_path', type=str, default=f'./NLP_CEFE/data/train.json')
    parser.add_argument('--valid_data_path', type=str, default=f'./NLP_CEFE/data/val.json')
    parser.add_argument('--conv_out_channels', type=int, default=50)
    parser.add_argument('--lstm_hid_dim', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=20)
    args = parser.parse_args()

    train_data = utils.load_data(args.train_data_path)
    val_data = utils.load_data(args.valid_data_path)
    print('Getting train data')
    train_data_processed = utils.process_data(train_data, LLM=True)
    print('Getting val data')
    val_data_processed = utils.process_data(val_data, LLM=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device).eval()

    train_layer_path = './NLP_CEFE/activations/train_layer_wise_act.pt'
    train_head_path  = './NLP_CEFE/activations/train_head_wise_act.pt'
    val_layer_path   = './NLP_CEFE/activations/val_layer_wise_act.pt'
    val_head_path    = './NLP_CEFE/activations/val_head_wise_act.pt'

    # 加载或计算训练集激活
    if os.path.exists(train_layer_path) and os.path.exists(train_head_path):
        train_data_processed['layer_wise_act'] = torch.load(train_layer_path)
        train_data_processed['head_wise_act']  = torch.load(train_head_path)
    else:
        train_sent_tokenized = tokenize_prompt(train_data_processed['sent_list'], tokenizer)
        train_data_processed['layer_wise_act'], train_data_processed['head_wise_act'] = get_activations(model, train_sent_tokenized, device)
        # torch.save(train_data_processed['layer_wise_act'], train_layer_path)
        # torch.save(train_data_processed['head_wise_act'], train_head_path)

    # 加载或计算验证集激活
    if os.path.exists(val_layer_path) and os.path.exists(val_head_path):
        val_data_processed['layer_wise_act'] = torch.load(val_layer_path)
        val_data_processed['head_wise_act']  = torch.load(val_head_path)
    else:
        val_sent_tokenized = tokenize_prompt(val_data_processed['sent_list'], tokenizer)
        val_data_processed['layer_wise_act'], val_data_processed['head_wise_act'] = get_activations(model, val_sent_tokenized, device)
        # torch.save(val_data_processed['layer_wise_act'], val_layer_path)
        # torch.save(val_data_processed['head_wise_act'], val_head_path)

    # print(len(train_data_processed['courseGrainedError_label_tensor']), len(train_data_processed['fineGrainedError_label_tensor']))
    all_pred_rst, all_probe = train_probe(train_data_processed)

    results = {} 

    top_k = [4, 8, 16, 32]
    threhold_list = [0.1 * ii for ii in range(1, 6)]
    best_combination = {'total_f1': 0}
    for k in top_k:
        top_k_heads, top_k_clf = get_top_heads(all_pred_rst, all_probe, k=k)

        f1_thresholds = []
        for threshold in threhold_list:
            course_result, fine_result = f1_cal(val_data_processed, top_k_heads, top_k_clf, threshold=threshold, k=k)
            f1, f1_course, f1_fine = utils.evaluate(torch.from_numpy(course_result), val_data_processed['courseGrainedError_label_tensor'],
                                                    torch.from_numpy(fine_result), val_data_processed['fineGrainedError_label_tensor'])
            if f1 > best_combination['total_f1']:
                best_combination['total_f1'] = f1
                best_combination['k'] = k
                best_combination['threshold'] = threshold
                best_combination['course_f1'] = f1_course
                best_combination['fine_f1'] = f1_fine
            print(f'k: {k}, threshold: {threshold:.1f}, total f1: {f1:.4f} course f1: {f1_course:.4f}, fine f1: {f1_fine:.4f}')
    
            f1_thresholds.append((threshold, f1))
        results[k] = f1_thresholds
    print("="*50)
    print(f'Best combination: k: {best_combination["k"]}, threshold: {best_combination["threshold"]:.1f}, total f1: {best_combination["total_f1"]:.4f}, course f1: {best_combination["course_f1"]:.4f}, fine f1: {best_combination["fine_f1"]:.4f}')
   
    plt.figure(figsize=(10, 6))

    for k in results:
        thresholds, f1_scores = zip(*results[k])
        plt.plot(thresholds, f1_scores, label=f'k={k}', marker='o')

    plt.title('F1 Score vs Threshold for Different k Values')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('f1_vs_threshold.png')
    plt.show()
    return

if __name__ == '__main__':
    main()