import math
from sklearn.svm import SVC
import jieba
jieba.setLogLevel(0)
import fasttext
from datasets import load_from_disk
import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np


def softmax1(x):
    total = sum([value for _, value in x])
    class_weight = {}
    for label, number in x:
        class_weight[label] = round(total / (2 * number), 2)
    return class_weight


def softmax2(x):
    total = sum([value for _, value in x])
    class_weight = {}
    for label, number in x:
        class_weight[label] = round(total / (total - number), 2)
    return class_weight


def do_train():

    train_data = load_from_disk('data/sentence.data')['train']
    y_true = train_data['label']
    embeddings = fasttext.load_model('pretrained/cc.zh.300.bin')
    inputs = []
    for data in train_data:
        title = data['title']
        label = data['label']
        title = ' '.join(jieba.lcut(title))
        title = embeddings.get_sentence_vector(title)
        inputs.append(title)

    class_distrib = sorted(Counter(y_true).items(), key=lambda x: x[1])
    print('样本分布:', class_distrib)
    class_weight = softmax1(class_distrib)
    class_weight = softmax2(class_distrib)
    class_weight = None
    print('样本权重:', class_weight)

    estimator = SVC(class_weight=class_weight, C=1, kernel='poly', degree=4)
    estimator.fit(inputs, y_true)
    y_pred = estimator.predict(inputs)

    print('标签顺序:', estimator.classes_)
    precision, recall, f_score, true_sum = precision_recall_fscore_support(y_true, y_pred)
    print('准确率:', accuracy_score(y_true, y_pred))
    print('精确率:', precision)
    print('召回率:', recall)
    print('F-score:', f_score)

    joblib.dump(estimator, 'finish/sentence/svm/svm.bin')


if __name__ == '__main__':
    do_train()