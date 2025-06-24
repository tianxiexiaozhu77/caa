import json
import re
from tqdm import tqdm
import jsonlines
import argparse
import numpy as np
import os
import time
from difflib import SequenceMatcher


# 获取命令行参数
def get_args():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='chatgpt')
    parser.add_argument("--dataset", type=str, default="GSM8K",
                        choices=["AQuA","GSM8K", "MultiA", "Addsub", "MathQA", "CSQA", "Strategyqa", "date_understanding", "MAWPS", "ASDIV", "SVAMP"])
    # parser.add_argument('--data_path', type=str, default=('/opt/data/private/zjx/ICL/I2CL-math/log/math/gsm8k_ours_test.json'))
    # parser.add_argument('--data_path', type=str, default=('/opt/data/private/zjx/ICL/I2CL-math/log/math/gsm8k_zero_shot-test.json'))
    # parser.add_argument('--data_path', type=str, default=('/opt/data/private/zjx/ICL/I2CL-math/log/math/test_llama_zeroshot.json'))
    # parser.add_argument('--data_path', type=str, default=('/opt/data/private/zjx/ICL/I2CL-math/log/math/gsm8k_zero_shot-test.json'))
    # parser.add_argument('--data_path', type=str, default='/opt/data/private/zjx/ICL/ICV_math/logger/main/gsm8k/seed0_default_llama-2_7b_zeroshot.json')
    # parser.add_argument('--data_path', type=str, default=('/opt/data/private/zjx/ICL/I2CL-math/log/math/test_llama_zeroshot_ceshi.json'))
    # parser.add_argument('--data_path', type=str, default=('/opt/data/private/zjx/ICV_math/logger/main/gsm8k/seed0_default_llama-2_7b_random20_icvstrength0.1_layer18_lasttoken.json'))
    # parser.add_argument('--data_path', type=str, default=('/opt/data/private/zjx/ICV_math/logger/main/gsm8k/seed0_default_llama-2_7b_random8_icvstrength0.2_layer18_lasttoken.json'))
    # parser.add_argument('--data_path', type=str, default=('/opt/data/private/zjx/ICV_math/logger/main/gsm8k/seed0_default_llama-2_7b_random40_icvstrength0.1_layer18_lasttoken.json'))
    parser.add_argument('--data_path', type=str, default=('/opt/data/private/zjx/ICV_math/logger/main/gsm8k/seed42_default_llama-2_7b_random4_icvstrength0.1_layer18_lasttoken.json'))
    # parser.add_argument('--data_path', type=str, default=('/opt/data/private/zjx/ICV_math/logger/main/gsm8k_ContrastSamplespos/seed0_default_llama-2_7b_random40_icvstrength0.2_layer18_lasttoken.json'))

    parser.add_argument('--type', type=str, default='all')
    parser.add_argument('--split', type=str, default='rest')
    parser.add_argument('--api_key', type=str, default='')
    # parser.add_argument('--type', type=str, default='')
    parser.add_argument('--iesc', default=False) # 是否开启Self-Consistency
    parser.add_argument('--sc_size', type=int) # 预测答案数量 
    args = parser.parse_args()  
    return args

# 计算预测文本与真实文本之间的精确匹配得分：统一小写后字符串完全匹配为1，否则为0
def exact_match_score(prediction, ground_truth): 
    if prediction is None or ground_truth is None:
        return 0
    numeric_re = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')
    # numeric_re = re.compile(r'^([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?|[+-]?\d+/\d+)$')
    if bool(numeric_re.match(prediction)) and bool(numeric_re.match(ground_truth)):
        if abs(float(prediction)-float(ground_truth)) < 0.001:
            return 1
        else :
            return 0
    elif isinstance(prediction, str) and isinstance(ground_truth, str):
        return round(SequenceMatcher(None, prediction, ground_truth).ratio())
    
    return int(prediction.strip().lower() == ground_truth.strip().lower())

'''
    计算给定文本的熵（用于衡量文本中单词分布的不确定性/多样性）：
    - 输入文本分割成单词列表,并转化为numpy数组,便于后续处理
    - 提取输入文本中不重复单词 x_value_list=set()
    - 初始化熵值ent = 0.0; 遍历每个不同单词x_value,计算其在文本中出现的概率p(即该单词出现次数除以总单词数); 然后计算概率对数并乘以概率p,得到该单词对总熵的贡献; 将贡献从总熵中减去; 直到得到熵值ent
    熵值越高，表示文本中的单词分布越均匀，信息量越大; 熵值越低，表示文本中的单词分布越集中，信息量越小
'''
def calc_ent(word):
    data = word.split()
    x = np.array(data)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent

# 计算预测结果和标准答案之间的F1score（precision和recall的调和平均）
def f1_score(prediction, ground_truth):
    # 预处理预测文本和真实文本：去除首尾空格，转换为小写，然后分割成单词列表。
    prediction_tokens = prediction.strip().lower().split()
    ground_truth_tokens = ground_truth.strip().lower().split()

    # 计算预测文本和真实文本中共有单词集合
    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)
    # 计算共有单词的数量：对于每个共有单词，计算它在预测文本和真实文本中出现的次数，取较小值然后求和——>避免重复计算多次出现的单词
    common_token_count = sum(
        [min(prediction_tokens.count(token), ground_truth_tokens.count(token)) for token in common_tokens])

    # 如果预测文本或真实文本为空（即没有单词），则直接比较两者是否相等;如果相等（都为空），返回1；否则返回0
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)

    # 计算精确率：共有单词数量除以预测文本的单词总数
    precision = 1.0 * common_token_count / len(prediction_tokens)
    # 计算召回率：共有单词数量除以真实文本的单词总数
    recall = 1.0 * common_token_count / len(ground_truth_tokens)

    # 如果精确率和召回率之和为0（即两者都为0），则F1分数为0——预测文本和真实文本完全不相交
    if precision + recall == 0:
        return 0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

'''
    旨在根据不同的数据集（dataset）类型对真实答案（truth）进行格式化和清洗
    GSM8K 数据集处理: 使用正则表达式提取出以 "So the final answer is " 开始，后面跟随数字、逗号或小数点的部分作为true_answer
    Addsub 数据集处理: 检查答案的小数部分是否为 "00" 或 "0"，如果是，则只保留整数部分,否则截断前6个字符作为true_answer
    date_understanding 数据集处理: 使用正则表达式查找所有出现的 A、B、C、D、E 或 F 字符，然后返回找到的最后一个字符
    MultiA 数据集处理: 直接将 truth 列表中的第一个元素转换为整数，然后转换回字符串并返回
    AQUA, Strategyqa, CSQA 数据集处理: 直接返回原始的 truth，不做任何处理。
'''
def true_answer_cleansing(args, truth):
    if args.dataset in ['GSM8K']:
        if args.split == "train":
            truth = re.search(r"####\s*(-?[\d,]+)", truth).group(1)
            return truth
        else:
            truth = re.findall("So the final answer is ([0-9|,|.]*)", truth)
        if not len(truth):
            print("ERROR! No truth answer. ", truth)
            return ""
        else:
            return truth[0].replace(',', '')

    if args.dataset in ['Addsub']:
        if truth[0].split(".")[-1] in ("00", "0"):
            truth = str(truth[0].split(".")[0])
            return truth
        if len(truth[0]) >= 6:
            truth[0] = truth[0][:6]
        return truth[0]
    if args.dataset in ["SVAMP"]:
        truth = str(truth)
        if truth.split(".")[-1] in ("00", "0"):
            truth = str(truth.split(".")[0])
            return truth
        if len(truth) >= 6:
            truth = truth[:6]
        return truth
    if args.dataset in ["MAWPS"]:
        truth = str(truth)
        if truth.split(".")[-1] in ("00", "0"):
            truth = str(truth.split(".")[0])
            return truth
        if len(truth) >= 6:
            truth = truth[:6]
        return truth
    
    if args.dataset in ["ASDIV"]:
        if " (" in truth:
            unit = truth.split(" (")[-1]
            truth = truth.split(" (")[0]

        if truth.split(".")[-1] in ("00", "0"):
            truth = str(truth.split(".")[0])
            return truth
        # if len(truth) >= 6:
        #     truth = truth[:6]
        return truth

    if args.dataset in ['date_understanding']:
        truth = re.findall(r'A|B|C|D|E|F', truth)
        return truth[-1]

    if args.dataset in ['MultiA']:
        return str(int(truth[0]))
        # return str(int(truth))

    if args.dataset in ['AQuA', 'Strategyqa', 'CSQA']:
        return truth

'''
    对模型的预测答案（pred）进行清洗和格式化，以便更准确地与真实答案进行比较或评估:
    - AQUA 和 CSQA 数据集处理: 使用正则表达式 re.findall 查找所有 A、B、C、D 或 E 字符,代表多项选择题的选项
    - date_understanding 数据集处理: 查找的选项范围扩展到 F
    - object_tracking 数据集处理: 只查找 A、B、C 三个选项
    - GSM8K, Addsub, MultiA 数据集处理: 首先移除所有逗号，然后使用正则表达式提取所有的数字（包括整数和小数，正负数都考虑在内）
    - Strategyqa 和 coin_flip 数据集处理: 首先将预测结果转换为小写，然后移除引号、换行、点号、空格、冒号和逗号，将结果分割成单词列表，最后只保留列表中的 "yes" 或 "no"
    通用后处理：如果处理后的 pred 是空列表，则将其设置为空字符串，否则取列表中的最后一个元素作为最终预测结果。
                对最终预测结果进行进一步的格式化：如果结果以句号结尾，去掉句号；如果结果的小数部分为 "00"，则只保留整数部分；移除所有空格。
'''
def pred_answer_cleansing(args, pred):
    if args.dataset in ["AQuA", "CSQA" , "date_understanding"]:
    
        match = re.search(r"the answer (.*)", pred, re.IGNORECASE)
        if match:
        # 从 "the answer is" 之后的文本中找到第一个 A, B, C, 或 D
            answer = match.group(1)
            match2 = re.search(r'\b([ABCDEF])\b', answer)
            if match2:
                return match2.group(1)
        # 如果没有找到符合条件的答案，返回空字符串
        else:
            res = re.findall(r'A|B|C|D|E|F', pred)
            if isinstance(res,list) and len(res):
                return str(res[-1])
            else :
                return  str(res)
        return ""
    elif args.dataset in ["GSM8K", "Addsub", "MultiA",]:  # "SVAMP"
        res=''
        # match1 = re.search(r"answer is (.*)", pred)
        # if match1 :
        #     answer = match1.group(1)
        #     match2 = re.search(r'\((.*?)\)', answer)
        #     if match2:
        #         res = match2.group(1)
        if len(res)==0 :
            pred = pred.replace(",", "")
            res = re.findall(r'-?\d+\.?\d*', pred)
        if isinstance(res,list) and len(res):
            return str(res[-1].rstrip("."))
        else :
            return  str(res)
        
    elif args.dataset in ["SVAMP"]:
        res=''
        # match1 = re.search(r"answer is (.*)", pred)
        # if match1 :
        #     answer = match1.group(1)
        #     match2 = re.search(r'\((.*?)\)', answer)
        #     if match2:
        #         res = match2.group(1)
        if len(res)==0 :
            pred = pred.replace(",", "")
            res = re.findall(r'-?\d+\.?\d*', pred)
        if isinstance(res,list) and len(res):
            return str(res[-1].split(" (")[-1].rstrip("."))
        elif not res:
            return "--"
        else:
            return  str(res.split(" (")[-1].rstrip("."))
        
    elif args.dataset in ["MAWPS"]:
        res=''
        # match1 = re.search(r"answer is (.*)", pred)
        # if match1 :
        #     answer = match1.group(1)
        #     match2 = re.search(r'\((.*?)\)', answer)
        #     if match2:
        #         res = match2.group(1)
        if len(res)==0 :
            pred = pred.replace(",", "")
            res = re.findall(r'-?\d+\.?\d*', pred)
        if isinstance(res,list) and len(res):
            return str(res[-1].split(" (")[-1].rstrip("."))
        else :
            return  str(res).split(" (")[-1].rstrip(".")
        
    elif args.dataset in ["ASDIV"]:
        res=''
        # match1 = re.search(r"answer is (.*)", pred)
        # if match1 :
        #     answer = match1.group(1)
        #     match2 = re.search(r'\((.*?)\)', answer)
        #     if match2:
        #         res = match2.group(1)
        if len(res)==0 :
            pred = pred.replace(",", "")
            res = re.findall(r'-?\d+\.?\d*', pred)
        if isinstance(res,list) and len(res):
            return str(res[-1]).split(" ")[0].rstrip(".")
        else :
            return  str(res).split(" ")[0].rstrip(".")
    elif args.dataset in ["Strategyqa"]:
        pred_all = pred.lower()
        start_index = pred_all.find("the answer is ")
        if start_index != -1:
            # 加上 len(phrase) 来从短语之后开始截取
            pred= pred_all[start_index + len("the answer is "):]
            if "yes" in pred:
                return "yes"
            elif "no" in pred:
                return "no"
            else:
                return pred
        # import pdb; pdb.set_trace()
        start_index = pred_all.find("the answer (yes or no) is ")
        if start_index != -1:
            pred= pred_all[start_index + len("the answer (yes or no) is "):]
            if "yes" in pred:
                return "yes"
            elif "no" in pred:
                return "no"
            else:
                return pred
            
        if "yes" in pred:
            return "yes"
        elif "no" in pred:
            return "no"
        else:
            return pred


# 计算模型在特定数据集上的准确率
def calcu_accuracy(args):
    
    path = args.data_path
    data = []
    truth_list = []
    pred_list = []
    with open(args.data_path, 'r', encoding='utf-8') as r_f:
        lines=r_f.readlines()
        for line in lines:
            d=eval(line)
            d["generation"].replace("\\n","\n")
            data.append(d)


    print()
    output_dir = os.path.dirname(args.data_path)
    question_path = os.path.join(os.path.dirname(output_dir),f"prompt/{args.dataset}_{args.type}_selectedQ.txt")
        
    selected_question=[]
    if os.path.exists(question_path):
        with open(question_path, 'r', encoding='utf-8') as f:
            questions = f.read()
            selected_question=questions.strip().split('\n\n\n')
    
    score = 0 # 用于记录模型预测正确的次数
    count = 0 # 用于记录数据集中总的样本数量
    # 遍历样本，格式化真实/预测答案，然后计算其exact_match_score
    
    
    process_num=5
    process_num_count=5
    done_num=0
    done_data={}

    idx_count=0
    data_to_test = 0
    for num_i, item in enumerate(tqdm(data)):
        # print(item)
        # if num_i > 1223:
        #     break
        data_to_test = data_to_test +1
        if "generation" not in item or "gold" not in item:
            print("ERROR", item)

        # if args.dataset in ['Addsub','date_understanding','MultiA','Strategyqa',"MAWPS","ASDIV","SVAMP"]:
        #     # print(item)
        #     if item['question'] in selected_question:
        #         continue
                # print(f"selected_question:{item['question']}")
            
        
        truth = true_answer_cleansing(args, item['gold']).lower()  # 真实答案清洗
        pred = pred_answer_cleansing(args, item['generation']).lower()  # 预测答案清洗
        
        em=0
        # 'MultiA',"GSM8K","Addsub"
        if args.dataset in []:
            res=done_data[idx_count]
            idx_count+=1
            if res=="YES":
                em=1
            else :
                em=0
        else:
            em = exact_match_score(pred, truth)

        # 错误日志记录：若exact_match_score=0（即预测错误），则记录错误信息（索引，真实/预测答案，对应prompt）
        global eval_path
        eval_path = "new_data/error_data/ablation/" + args.dataset + "_" + args.type + "_" + args.model + "_"+ args.split + "_error.log"
        error_data_path = "new_data/error_data/ablation/" + args.dataset + "_" + args.type + "_"+ args.model  + "_"+ args.split + ".jsonl"
        file_dir = os.path.dirname(eval_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
            
        if em <= 0.5:
            pass
            # with open(eval_path, "a", encoding='utf-8') as f:
            #     f.write("idx:" + str(count) + "\n" + str(item['gold']) + "\n" + item['generation'] + "\nTruth:" + truth + "\nPred:" + pred + "\n\n")
            # with jsonlines.open(error_data_path, 'a') as f:
            #     res_dict = {}
            #     res_dict['prompt'] = item['generation']
            #     res_dict['true_answer'] = item['gold']
            #     f.write(res_dict)
        # 更新评分和计数器
        score += em
        count += 1

    print(f"final acc:{score / data_to_test}")    
    # 打印并记录最终准确率
    print(args.data_path)
    print(f"final acc:{score / len(data)}")
    with open(eval_path, "a", encoding='utf-8') as f:
        f.write(
            "\ncorrect_number:{} \ntotal_number: {} \ncalcu_accuracy:{} \n".format(score, len(data), score / len(data)))

    return score / len(data)



'''
    计算模型在多个预测结果中自洽性的准确率:
    - 情景：模型对同一个问题生成多个不同的答案，并且需要确定哪个答案是最可能正确的
    - 计算方式：比较多个预测答案，并选择最常出现的答案作为最终答案；若启用args.iesc，函数还会根据问题的信息熵来确定考虑的预测答案数量
'''


if __name__ == '__main__':
    # import pdb;pdb.set_trace()
    args = get_args()
    accuracy = calcu_accuracy(args)
    # accuracy = calcu_self_cons_accuracy(args)


