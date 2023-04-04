import datasets
datasets.disable_progress_bar()
from transformers import BertTokenizer
from transformers import AlbertForTokenClassification
from transformers import BertForTokenClassification
from datasets import load_from_disk
from aspect_extraction import parse_labels
import glob
import torch
import numpy as np
import re


@torch.no_grad()
def do_prediction(estimator, tokenizer, test_data):

    pred_labels = []
    def function(batch_data):

        tokens = [tokens for tokens in batch_data['tokens']]
        inputs = tokenizer(tokens, padding='longest', add_special_tokens=False, return_token_type_ids=False, return_tensors='pt')
        logits = estimator(**inputs).logits
        labels = torch.argmax(logits, dim=-1)
        # labels是padding之后，通过掩码确认每一个样本实际的labels长度
        for label, mask in zip(labels, inputs.attention_mask):
            label = label[:torch.sum(mask)]
            pred_labels.append(label.tolist())
    test_data.map(function, batched=True, batch_size=32)

    return pred_labels


@torch.no_grad()
def do_prediction_paddle(estimator, tokenizer, test_data):

    pred_labels = []
    def function(batch_data):

        tokens = [tokens for tokens in batch_data['tokens']]
        inputs = tokenizer(tokens, padding='longest', add_special_tokens=False, return_token_type_ids=False, return_tensors='pt')
        logits = estimator(**inputs).logits
        labels = torch.argmax(logits, dim=-1)
        # labels是padding之后，通过掩码确认每一个样本实际的labels长度
        for label, mask in zip(labels, inputs.attention_mask):
            label = label[:torch.sum(mask)]
            pred_labels.append(label.tolist())
    test_data.map(function, batched=True, batch_size=32)

    return pred_labels


def do_evaluation():

    # 模型路径
    model_names = glob.glob('finish/aspect/albert/epoch-*')
    # 测试数据
    test_data = load_from_disk('data/aspect.data')['test']
    # 真实标签
    true_labels = [data['labels'] for data in test_data]
    true_tokens = [''.join(data['tokens']) for data in test_data]

    for model_name in model_names:

        print('模型名字:', model_name)
        # estimator = BertForTokenClassification.from_pretrained(model_name, num_labels=5).eval()
        estimator = AlbertForTokenClassification.from_pretrained(model_name, num_labels=5).eval()
        tokenizer = BertTokenizer.from_pretrained(model_name)
        # pred_labels = do_prediction(estimator, tokenizer, test_data)
        pred_labels = do_prediction_paddle(estimator, tokenizer, test_data)

        all_true_labels = sum(true_labels, [])
        all_pred_labels = sum(pred_labels, [])

        # 计算token的预测正确率
        # all_pred_label_number 预测正确标签数量
        # all_true_label_number 总标签数量
        all_pred_label_number = (np.array(all_true_labels) == np.array(all_pred_labels)).sum()
        all_true_label_number = len(all_true_labels)
        print('token 准确率: %d/%d' % (all_pred_label_number, all_true_label_number))
        # 计算所有aspect和opinion词
        all_true_labels_string = ''.join(map(str, all_true_labels))
        all_pred_labels_string = ''.join(map(str, all_pred_labels))

        aspect_total, aspect_right, aspect_wrong = 0, 0, 0
        opinion_total, opinion_right, opinion_wrong = 0, 0, 0
        for postision in re.finditer(r'12*|34*', all_true_labels_string):
            # 获得真实观点情感词位置
            true_start, true_end = postision.start(), postision.end()

            # 表示当前词为 aspect
            if all_true_labels_string[true_start] == '1':
                aspect_total += 1
                if all_true_labels_string[true_start: true_end] == all_pred_labels_string[true_start: true_end]:
                    aspect_right += 1
                else:
                    aspect_wrong += 1

            # 表示当前词为 opinion
            if all_true_labels_string[true_start] == '3':
                opinion_total += 1
                if all_true_labels_string[true_start: true_end] == all_pred_labels_string[true_start: true_end]:
                    opinion_right += 1
                else:
                    opinion_wrong += 1

        print('aspec 准确率: %d/%d' % (aspect_right, aspect_total))
        print('opini 准确率: %d/%d' % (opinion_right, opinion_total))
        print('-' * 100)


if __name__ == '__main__':
    do_evaluation()