from sklearn.svm import SVC
import jieba
jieba.setLogLevel(0)
import fasttext
from datasets import load_from_disk
import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


def do_evaluation():

    test_data = load_from_disk('data/sentence.data')['test']
    embeddings = fasttext.load_model('pretrained/cc.zh.300.bin')
    estimator = joblib.load('finish/sentence/svm/svm.bin')
    y_true = test_data['label']
    inputs = []
    for data in test_data:
        title = data['title']
        label = data['label']
        title = ' '.join(jieba.lcut(title))
        title = embeddings.get_sentence_vector(title)
        inputs.append(title)

    y_pred = estimator.predict(inputs)

    precision, recall, f_score, true_sum = precision_recall_fscore_support(y_true, y_pred)
    print('准确率:', accuracy_score(y_true, y_pred))
    print('精确率:', precision)
    print('召回率:', recall)
    print('F-score:', f_score)


if __name__ == '__main__':
    do_evaluation()

    import fastNLP.models.torch.sequence_labeling