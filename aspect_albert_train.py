import torch.cuda
from transformers import AlbertForTokenClassification
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from datasets import load_from_disk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def do_train(estimator, tokenizer, save_folder):

    def collate_function(batch_data):
        tokens, labels = [], []
        for data in batch_data:
            tokens.append(data['tokens'])
            labels.append(torch.tensor(data['labels'], device=device))
        tokens = tokenizer.batch_encode_plus(tokens, padding='longest', return_token_type_ids=False, add_special_tokens=False, return_tensors='pt')
        tokens = {key: value.to(device) for key, value in tokens.items()}
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return tokens, labels

    traindata = load_from_disk('data/aspect.data')['train']
    dataloader = DataLoader(traindata, batch_size=64, shuffle=True, collate_fn=collate_function)
    optimizer = optim.Adam(estimator.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epoch_number = 100
    scheduler = ReduceLROnPlateau(optimizer, factor=0.7, patience=1, cooldown=0, verbose=False)

    for epoch in range(1, epoch_number + 1):
        progress = tqdm(range(len(dataloader)), ncols=110)
        total_loss = 0.0
        total_number, total_right = 0, 0
        for tokens, labels in dataloader:
            # 模型计算
            outputs = estimator(**tokens)
            loss = criterion(outputs.logits.reshape(-1, 5), labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算训练集 token 准确率
            y_pred = torch.argmax(outputs.logits, dim=-1)
            mask = labels != -100
            labels = torch.masked_select(labels, mask)
            y_pred = torch.masked_select(y_pred, mask)
            total_right += (y_pred == labels).sum()
            total_number += len(labels)
            # 打印训练信息
            total_loss += loss.item() * (labels != -100).sum()
            progress.set_description('epoch %3d/%d loss %12.6f lr %.8f acc %4d/%4d' % (epoch, epoch_number, total_loss, scheduler.optimizer.param_groups[0]['lr'], total_right, total_number))
            progress.update()
            break

        scheduler.step(total_loss.item())
        progress.close()

        if epoch % 5 == 0:
            tokenizer.save_pretrained('finish/aspect/' + save_folder + '/epoch-%d-%.4f' % (epoch, total_loss))
            estimator.save_pretrained('finish/aspect/' + save_folder + '/epoch-%d-%.4f' % (epoch, total_loss))


def train_start():

    # 加载模型
    checkpoint = 'pretrained/albert_chinese_tiny'
    estimator = AlbertForTokenClassification.from_pretrained(checkpoint, num_labels=5).to(device)
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    # 开始训练
    do_train(estimator, tokenizer, 'albert')


if __name__ == '__main__':
    train_start()




