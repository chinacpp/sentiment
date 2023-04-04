from transformers import AlbertForSequenceClassification
from transformers import BertTokenizer
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def softmax1(x):
    total = sum([value for _, value in x])
    class_weight = []
    for label, number in x:
        class_weight.append(round(total / (2 * number), 2))
    return class_weight


def do_train():

    train_data = load_from_disk('data/sentence.data')['train']
    estimator = AlbertForSequenceClassification.from_pretrained('pretrained/albert_chinese_tiny', num_labels=2).to(device)
    tokenizer = BertTokenizer.from_pretrained('pretrained/albert_chinese_tiny')
    optimizer = optim.Adam(estimator.parameters(), lr=1e-5)

    class_weight = torch.tensor([1.6, 0.73], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    def collate_function(batch_data):
        inputs = [data['title'] for data in batch_data]
        labels = [data['label'] for data in batch_data]
        inputs = tokenizer(inputs, padding='longest', return_token_type_ids=False, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = torch.tensor(labels, device=device)
        return inputs, labels

    dataloader = DataLoader(train_data, shuffle=True, batch_size=32, collate_fn=collate_function)
    for epoch in range(25):

        progress = tqdm(range(len(dataloader)))
        total_right, total_number, total_loss = 0, 0, 0.0
        for inputs, labels in dataloader:
            output = estimator(**inputs)
            loss = criterion(output.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred = torch.argmax(output.logits, dim=-1)
            total_right += (y_pred == labels).sum()
            total_number += len(labels)
            total_loss += loss.item()

            desc = 'epoch %2d loss %8.4f acc %4d/%4d' % (epoch + 1, total_loss, total_right, total_number)
            progress.set_description(desc)
            progress.update()
        progress.close()

        save_path = 'finish/sentence/albert/epoch-%d-loss-%.4f' % (epoch+1, total_loss)
        estimator.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


if __name__ == '__main__':
    do_train()