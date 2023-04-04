import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import paddlenlp.data
import pandas as pd
from paddlenlp.transformers import SkepForTokenClassification
from paddlenlp.transformers import SkepTokenizer
from torch.utils.data import DataLoader
import paddle
from paddlenlp.data import Pad
from tqdm import tqdm
import paddle.nn as nn
import paddle.optimizer as optim
import glob
import numpy as np
from datasets import load_from_disk


paddle.set_device('cpu')


def do_train():

    checkpoint = 'pretrained/skep_ernie_1.0_large_ch'
    estimator = SkepForTokenClassification.from_pretrained(checkpoint, num_classes=5)
    tokenizer = SkepTokenizer.from_pretrained(checkpoint)

    def collate_function(batch_data):
        inputs = [data['tokens'] for data in batch_data]
        labels = [data['labels'] for data in batch_data]
        inputs = tokenizer.batch_encode(inputs, is_split_into_words=True, padding='longest', add_special_tokens=False, return_tensors='pd')
        labels = paddle.to_tensor(Pad(pad_val=-100)(labels))
        return inputs, labels

    traindata = load_from_disk('data/aspect.data')['train']
    dataloader = DataLoader(traindata, batch_size=32, shuffle=True, collate_fn=collate_function)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(parameters=estimator.parameters(), learning_rate=2e-5)

    for epoch in range(1, 16):

        progress = tqdm(range(len(dataloader)), ncols=100)
        total_loss, total_size = 0.0, 0.0
        total_number, total_right = 0, 0
        for inputs, labels in dataloader:
            outputs = estimator(**inputs)
            loss = criterion(outputs, labels)
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            y_pred = paddle.argmax(outputs, axis=-1)
            mask = labels != -100
            labels = paddle.masked_select(labels, mask)
            y_pred = paddle.masked_select(y_pred, mask)
            total_right += (y_pred == labels).sum()
            total_number += len(labels)
            total_loss += loss.item()
            total_size += len(labels)
            progress.set_description('epoch %2d loss %7.4f acc %4d/%4d' % (epoch, total_loss, total_right, total_number))
            progress.update()
        progress.close()

        tokenizer.save_pretrained('finish/aspect/skep/epoch-%d-%.4f' % (epoch, total_loss))
        estimator.save_pretrained('finish/aspect/skep/epoch-%d-%.4f' % (epoch, total_loss))


if __name__ == '__main__':
    do_train()




