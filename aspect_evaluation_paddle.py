import paddle
from paddlenlp.transformers import SkepTokenizer
from paddlenlp.transformers import SkepForTokenClassification
import glob
from datasets import load_from_disk
import numpy as np
import re


@paddle.no_grad()
def do_prediction(estimator, tokenizer, test_data):

    pred_labels = []
    def function(batch_data):

        tokens = [tokens for tokens in batch_data['tokens']]
        inputs = tokenizer.batch_encode(tokens, is_split_into_words=True, padding='longest', return_token_type_ids=False, return_attention_mask=True, add_special_tokens=False, return_tensors='pd')
        logits = estimator(inputs['input_ids'])
        labels = paddle.argmax(logits, axis=-1)
        # labels是padding之后，通过掩码确认每一个样本实际的labels长度
        for label, mask in zip(labels, inputs['attention_mask']):
            label = label[:paddle.sum(mask)]
            pred_labels.append(label.tolist())
    test_data.map(function, batched=True, batch_size=4)

    return pred_labels


def do_evaluation():

    # 模型路径
    checkpoints = glob.glob('finish/aspect/skep/epoch-*')
    # 测试数据
    test_data = load_from_disk('data/aspect.data')['test']
    # 真实标签
    true_labels = [data['labels'] for data in test_data]
    true_tokens = [''.join(data['tokens']) for data in test_data]

    for checkpoint in checkpoints:

        print('模型名字:', checkpoint)
        estimator = SkepForTokenClassification.from_pretrained(checkpoint)
        estimator.eval()
        tokenizer = SkepTokenizer.from_pretrained(checkpoint)
        pred_labels = do_prediction(estimator, tokenizer, test_data)

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