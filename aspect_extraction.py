import torch
from transformers import BertForTokenClassification
from transformers import AlbertForTokenClassification
from transformers import BertTokenizer
import glob
from datasets import load_from_disk
import numpy as np
import re
import pickle


def parse_labels(labels, sentence):

    if len(labels) != len(sentence):
        raise ValueError("len(labels) != len(sentence)")

    # 标签映射为数字
    label_to_index = pickle.load(open('data/label_to_index.pkl', 'rb'))
    # labels:          [0, 0, 1, 2, 0, 3, 4, 1, 2, 0, 0, 3]
    # non_zero_indexs: [2, 3, 5, 6, 7, 8, 11]
    # 获得非0标签值的位置
    non_zero_indexs = np.nonzero(labels)[0].tolist()
    # non_zero_labels = [1, 2, 3, 4, 1, 2, 3]
    # 获得真实的非0标签序列
    non_zero_labels = [labels[index] for index in non_zero_indexs]
    # index_labels = 1234123  转字符串，便于正则匹配
    non_zero_labels_string = ''.join(map(str, non_zero_labels))

    # 提取所有的标签及其类型
    words, types = [], ''
    for position in re.finditer(r'(12*)|(34*)', non_zero_labels_string):
        # 获得在index_labels中的开始和结束位置
        start, end = position.start(), position.end()
        # 存储当前词的类型，是aspect还是opinion
        types += non_zero_labels_string[start]
        # 获得在labels_index中的开始和结束位置
        start, end = non_zero_indexs[start], non_zero_indexs[end-1]
        words.append((sentence[start: end+1]))

    return types, words


def combine_aspect_opinion(types, words):

    # 优先匹配左侧的观点，如果右侧不存在的话，就匹配右侧的
    results = {}
    for position in re.finditer(r'(3+1)|(13+)', types):
        aspect = None
        opinions = []
        for i in range(position.start(), position.end()):
            if types[i] == '1':  # 判断是否是 aspect
                aspect = words[i]
            else:
                opinions.append(words[i])
        if aspect is not None:
            results[aspect] = opinions

    return results


@torch.no_grad()
def test():

    model_name = glob.glob('finish/aspect/albert/epoch-102-*')[0]
    print('模型:', model_name)
    estimator = AlbertForTokenClassification.from_pretrained(model_name, num_labels=5).eval()
    tokenizer = BertTokenizer.from_pretrained(model_name)

    aspect_data = load_from_disk('data/aspect.data')['test']
    index = 7
    tokens, labels = aspect_data[index]['tokens'], aspect_data[index]['labels']
    sentence = ''.join(tokens.split())
    print('句子:', sentence)
    print('-' * 55)

    print('真实:', labels)
    types, words = parse_labels(labels, sentence)
    print(types, words)
    print('观点:', combine_aspect_opinion(types, words))
    print('-' * 55)

    inputs = tokenizer([tokens], add_special_tokens=False, return_token_type_ids=False, return_tensors='pt')
    output = estimator(**inputs)
    labels = torch.argmax(output.logits, axis=-1).squeeze().tolist()

    print('预测:', labels)
    types, words = parse_labels(labels, sentence)
    print(types, words)
    print('观点:', combine_aspect_opinion(types, words))


if __name__ == '__main__':
    test()