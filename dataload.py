import json
import jsonlines
import os,re
import pickle
# from tools import read_jsonl, answer2jsonl
import pandas as pd

'''
    dataload class 封装用于加载和处理特定dataset: 模块化/可重用
    - 数据集包括: GSM8K、AQuA、MultiA、Strategyqa、Date_understanding、CSQA、Addsub 
    - 提供访问单个问题及其相关答案，以及确定每个数据集中条目总数的功能
        - __init__(data_path): 基于给定路径，读入并初始化数据
        - get_question_by_idx(idx): 基于索引获取 question/answer/rationale/options/Solutions等具体数据内容(dataset specific)
        - get_data_length(): 获取数据集大小
'''

class GSM8K(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for data in jsonlines.Reader(r_f):
                self.data.append(data)

    def get_question_by_idx(self, idx):
        question = self.data[idx]['question']
        answer = self.data[idx]['answer']
        return question, answer

    def get_data_length(self):
        return len(self.data)


class AQuA(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for data in jsonlines.Reader(r_f):
                self.data.append(data)
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx):
        question = self.data[idx]['question']
        answer = self.data[idx]['correct']
        cot = self.data[idx]['rationale']
        option = self.data[idx]['options']
        return question, answer, cot, option

    def get_data_length(self):
        return len(self.data)


class MultiA(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for data in jsonlines.Reader(r_f):
                self.data.append(data)
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx):
        question = self.data[idx]['sQuestion']
        answer = self.data[idx]['lSolutions']
        return question, answer

    def get_data_length(self):
        return len(self.data)


class Strategyqa(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            self.data = json.load(f)['examples']
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx):
        question = self.data[idx]['input']
        answer = 'Yes' if self.data[idx]['target_scores']['Yes'] else 'No'
        return question, answer

    def get_data_length(self):
        return len(self.data)


class Date_understanding(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            self.data = json.load(f)['examples']
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx):
        
        parts = self.data[idx]['input'].split("Options:\n")
        question = parts[0].strip()
        option = "Options:  "+"  ".join(parts[1].split('\n'))
        match = re.search(r"\((\w)\)", self.data[idx]['target'])
        answer = match.group(1)
        return question, answer, option

    def get_data_length(self):
        return len(self.data)


class CSQA(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                self.data.append(data)
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx):
        question = self.data[idx]['question']['stem']
        answer = self.data[idx]['answerKey']
        op = "Options: "
        for i in self.data[idx]['question']['choices']:
            op = op + " (" + i['label'] + ") " + i['text'] + " "
        return question, answer, op

    def get_data_length(self):
        return len(self.data)


class Addsub(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for data in jsonlines.Reader(r_f):
                self.data.append(data)

    def get_question_by_idx(self, idx):
        question = self.data[idx]['sQuestion']
        answer = self.data[idx]['lSolutions']
        return question, answer

    def get_data_length(self):
        return len(self.data)

class MAWPS(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for data in jsonlines.Reader(r_f):
                self.data.append(data)
                # print(data['target'])

    def get_question_by_idx(self, idx):
        question = self.data[idx]['input']
        answer = self.data[idx]['target']
        return question, answer

    def get_data_length(self):
        return len(self.data)

class ASDIV(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for data in jsonlines.Reader(r_f):
                self.data.append(data)

    def get_question_by_idx(self, idx):
        question = self.data[idx]['body'] + " " + self.data[idx]['question']
        answer = self.data[idx]['answer']
        return question, answer

    def get_data_length(self):
        return len(self.data)
    
class SVAMP(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as r_f:
            for data in jsonlines.Reader(r_f):
                self.data.append(data)

    def get_question_by_idx(self, idx):
        question = self.data[idx]['Body'].rstrip(".") + ". " + self.data[idx]['Question']
        answer = self.data[idx]['Answer']
        return question, answer

    def get_data_length(self):
        return len(self.data)

class CSQA(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                self.data.append(data)
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx):
        question = self.data[idx]['question']['stem']
        answer = self.data[idx]['answerKey']
        op = "Options: "
        for i in self.data[idx]['question']['choices']:
            op = op + " (" + i['label'] + ") " + i['text'] + " "
        return question, answer, op

class Date_understanding(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            self.data = json.load(f)['examples']
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx):
        
        parts = self.data[idx]['input'].split("Options:\n")
        question = parts[0].strip()
        option = "Options:  "+"  ".join(parts[1].split('\n'))
        match = re.search(r"\((\w)\)", self.data[idx]['target'])
        answer = match.group(1)
        return question, answer, option

    def get_data_length(self):
        return len(self.data)

class Strategyqa(object):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            self.data = json.load(f)['examples']
        # import pdb; pdb.set_trace()

    def get_question_by_idx(self, idx):
        question = self.data[idx]['input']
        answer = 'Yes' if self.data[idx]['target_scores']['Yes'] else 'No'
        return question, answer

    def get_data_length(self):
        return len(self.data)

# 用于处理错误数据(error_data)，数据格式{'prompt':prompt, 'true_answer':true_answer}
class ErrorData(object):
    def __init__(self, name):
        data_path = "data\\error_data\\" + name + "_" + "davinci003.jsonl"
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            for data in jsonlines.Reader(f):
                self.data.append(data)

    def get_question_by_idx(self, idx):
        question = self.data[idx]['prompt']
        answer = self.data[idx]['true_answer']
        return question, answer

    def get_data_length(self):
        return len(self.data)

# 通用数据集类，根据提供的数据集名称和路径，动态创建相应数据集对象。
class Dataset(object):
    def __init__(self, data_name, data_path) -> None:
        self.data = data_path
        self.name = data_name
        self.dataclass = self.get_dataclass()

    # 使用get_dataclass方法根据数据集名称决定使用哪个具体的数据集类
    def get_dataclass(self):
        if self.data is None:
            return ErrorData(self.name)
        if self.name in ['gsm8k']:
            return GSM8K(self.data)
        elif self.name in ['aqua']:
            return AQuA(self.data)
        elif self.name in ['multia']:
            return MultiA(self.data)
        elif self.name in ['Strategyqa']:
            return Strategyqa(self.data)
        elif self.name in ['date_understanding']:
            return Date_understanding(self.data)
        elif self.name in ['addsub']:
            return Addsub(self.data)
        elif self.name in ['CSQA']:
            return CSQA(self.data)
        elif self.name in ['mawps']:
            return MAWPS(self.data)
        elif self.name in ['asdiv']:
            return ASDIV(self.data)
        elif self.name in ['svamp']:
            return SVAMP(self.data)
        elif self.name in ['csqa']:
            return CSQA(self.data)
        elif self.name in ['data_un']:
            return Date_understanding(self.data)
        elif self.name in ['strategyqa']:
            return Strategyqa(self.data)
        else:
            raise NotImplementedError
