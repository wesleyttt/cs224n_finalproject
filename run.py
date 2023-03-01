import torch
import torch.nn as nn
from tqdm import tqdm
import json
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split

from model import Seq2SeqModel
from data_processing import read_data
from lyricsdataset import LyricsDataset

# replace these with strings of our desired HF transformer
encoder = None
decoder = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_path = "model.params"

lyrics, annotations = read_data.read_data()
lyrics_train, lyrics_test, annotations_train, annotations_test = train_test_split(lyrics, annotations, test_size=0.31)


train_dataset = LyricsDataset(lyrics_train, annotations_train)
test_dataset = LyricsDataset(lyrics_test, annotations_test)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

model = Seq2SeqModel("roberta-base")
model = model.to(device)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train():
    epochs = 10
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for batch_idx, (input_ids, attention_mask, output_ids) in tqdm(enumerate(train_dataloader)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output_ids = output_ids.to(device)

            # Set gradients to zero
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=output_ids[:, :-1])[0]

            # Compute loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), output_ids[:, 1:].contiguous().view(-1))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print statistics
        print('Epoch [%d]/[%d]\tTraining Loss: %.3f' % (epoch + 1, epochs, running_loss / len(train_dataloader)))

    model.save(model_save_path)
    torch.save(optimizer.state_dict(), model_save_path + '.optim')


def evaluate():
    model = torch.load(model_save_path)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, output_ids) in enumerate(test_dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output_ids = output_ids.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=output_ids[:, :-1])[0]

            # Compute loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), output_ids[:, 1:].contiguous().view(-1))
            total_loss += loss.item()

    print('Test Loss: %.3f' % (total_loss / len(test_dataloader)))


if __name__ == "__main__":
    train()
    evaluate()
