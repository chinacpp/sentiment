import torch
from transformers import AlbertForSequenceClassification
from transformers import BertTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import glob
from datasets import load_from_disk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def do_prediction(estimator, tokenizer, test_data):

    labels = []
    def function(batch_data):
        inputs = batch_data['title']
        inputs = tokenizer(inputs, padding='longest', return_token_type_ids=False, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = estimator(**inputs)
        y_pred = torch.argmax(output.logits, dim=-1)
        labels.extend(y_pred.tolist())
    test_data.map(function, batched=True, batch_size=32)

    return labels


def do_evaluation():

    checkpoints = glob.glob('finish/sentence/albert/epoch*')
    test_data = load_from_disk('data/sentence.data')['test']
    y_true = test_data['label']
    for checkpoint in checkpoints:
        print('模型:', checkpoint)
        estimator = AlbertForSequenceClassification.from_pretrained(checkpoint).eval().to(device)
        tokenizer = BertTokenizer.from_pretrained(checkpoint)
        y_pred = do_prediction(estimator, tokenizer, test_data)

        precision, recall, f_score, true_sum = precision_recall_fscore_support(y_true, y_pred)
        print('准确率:', accuracy_score(y_true, y_pred))
        print('精确率:', precision)
        print('召回率:', recall)
        print('F-score:', f_score)
        print('-' * 50)


if __name__ == '__main__':
    do_evaluation()