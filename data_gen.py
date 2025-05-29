import concurrent.futures
import json
import requests
from tqdm import tqdm
from json.decoder import JSONDecodeError

url = "https://api.siliconflow.cn/v1/chat/completions"
api = "sk-fffbhujadiuznntfevfnfouukadasecmkzizzbmtggsbfpxb"

def llm_sent_gen(content):
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {
                "content": content,
                "role": "user"
            }
        ],
        "response_format": {"type": "text"}
    }
    headers = {
        "Authorization": f"Bearer {api}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    response = response.json()
    response = response['choices'][0]['message']['content']
    try:
        response = json.loads(response)
    except JSONDecodeError:
        response = json.loads('{\"sent\": "error"}')
    return response



def sent_gen(content, num):
    sent_list = []
    inputs = [content] * num
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, num)) as executor:
        # 使用 executor.map 方法并行执行 llm_essay_gen 函数
        results = list(tqdm(executor.map(llm_sent_gen, inputs), total=num, unit='sent'))
        sent_list.extend(results)
    return sent_list


if __name__ == '__main__':
    error_type = ["缺字漏字", "错别字错误", "缺少标点", "错用标点",
                  "主语不明", "谓语残缺", "宾语残缺", "其他成分残缺",
                  "主语多余", "虚词多余", "其他成分多余",
                  "语序不当", "动宾搭配不当", "其他搭配不当"]
    gen_num = [44, 9, 23, 9, 29, 30, 32, 32, 32, 31, 22, 28, 28, 34]
    with open('./NLP_CEFE/prompt.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    rst = []
    for error, num in zip(error_type, gen_num):
        new_content = content + error
        error_rst = sent_gen(new_content, num)
        rst.extend(error_rst)

    try:
        with open('./NLP_CEFE/new_data.json', 'w', encoding='utf-8') as f:
            json.dump(rst, f, ensure_ascii=False, indent=2)
        print("JSON 文件保存成功！")
    except Exception as e:
        print(f"保存文件时出错: {e}")