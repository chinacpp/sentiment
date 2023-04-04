import pandas as pd
import glob
from datasets import Dataset
import codecs
from pyhanlp import JClass
import datasets
datasets.disable_progress_bar()
import pickle


def read_csv(csv_path):

    tokens, labels = [], []
    for line in codecs.open(csv_path):
        token, label = line.split('\t')
        tokens.append(token)
        labels.append(label)

    return tokens, labels


def aspect_data_process():

    all_tokens, all_labels = [], []
    for csv_path in glob.glob('corpus/aspect/*.txt'):
        tokens, labels = read_csv(csv_path)
        all_tokens.extend(tokens)
        all_labels.extend(labels)


    # 标签映射为数字
    label_to_index = {label.strip(): index for index, label in enumerate(open('corpus/aspect/label_ext.dict'))}
    pickle.dump(label_to_index, open('data/label_to_index.pkl', 'wb'))

    # 句子分割成单字，并转换标签为数字
    token_list, label_list = [], []
    for tokens, labels in zip(all_tokens, all_labels):

        tokens = list(tokens)
        labels = [label_to_index[label] for label in labels.split()]

        if len(tokens) != len(labels):
            continue

        # 去掉空格字符及其对应的标签
        my_tokens, my_labels = [], []
        for token, label in zip(tokens, labels):
            if token == ' ':
                continue
            my_tokens.append(token)
            my_labels.append(label)


        token_list.append(' '.join(my_tokens))
        label_list.append(my_labels)


    # 分割数据集
    aspect_data = Dataset.from_dict({'tokens': token_list, 'labels': label_list})
    aspect_data = aspect_data.train_test_split(test_size=0.1)
    aspect_data.save_to_disk('data/aspect.data')
    print(aspect_data)


normalizer = JClass('com.hankcs.hanlp.dictionary.other.CharTable')

def clean_commtent(comment):

    comment = comment.strip()
    comment = normalizer.convert(comment)
    comment = comment[:500]
    comment_length = len(comment)
    # 去除两侧标点符号
    quotation_marks = list('「」『』\'"!@#$%^&*()[];<>~￥……（）-+=《》、/')
    start, end = 0, comment_length-1
    start_over, end_over = False, False
    for start in range(comment_length):
        if comment[start] not in quotation_marks:
            break
    for end in range(comment_length-1, -1, -1):
        if comment[end] not in quotation_marks:
            break
    comment = comment[start: end + 1]

    return comment, len(comment) > 5


def comments_data_process():

    lines = codecs.open('corpus/sentence/comments.csv')
    lines.readline()
    titles, labels = [], []
    for line in lines:
        pos = line.find(',')
        label, title = line[:pos], line[pos + 1:]
        title, flag = clean_commtent(title)
        if not flag:
            continue
        titles.append(title)
        labels.append(int(label))

    sentence_data = Dataset.from_dict({'title': titles, 'label': labels})
    sentence_data = sentence_data.train_test_split(test_size=0.1)
    sentence_data.save_to_disk('data/sentence.data')
    print(sentence_data)



def aspect_math_data_process():
    pass



if __name__ == '__main__':
    aspect_data_process()
    comments_data_process()
