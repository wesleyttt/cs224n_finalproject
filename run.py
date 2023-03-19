import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from sklearn.model_selection import train_test_split

from data_processing import read_data
from lyricsdataset import LyricsDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_path = "model.params"

lyrics, annotations = read_data.read_data()
lyrics_train, lyrics_test, annotations_train, annotations_test = train_test_split(lyrics, annotations, test_size=0.31)

train_dataset = LyricsDataset(lyrics_train, annotations_train)
test_dataset = LyricsDataset(lyrics_test, annotations_test)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

model = T5ForConditionalGeneration.from_pretrained("t5-small")
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id).cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train():
    epochs = 10
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        losses = []
        for batch_idx, data in pbar:
            y = data['target_ids'].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]

            if batch_idx % 500 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_description(f"epoch {epoch + 1} iter {batch_idx}: train loss {loss.item():.5f}.")

        # Print statistics
        print('Epoch [%d]/[%d]\tTraining Loss: %.3f' % (epoch + 1, epochs, running_loss / len(train_dataloader)))

        # save progress for every epoch
        torch.save(model.state_dict(), model_save_path)
        torch.save(model.state_dict(), model_save_path + ".optim")

    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), model_save_path + '.optim')


def evaluate():
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    total_loss = 0.0
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(input_ids=ids, attention_mask=mask, max_length=512, num_beams=2,
                                           early_stopping=True, no_repeat_ngram_size=2)
            preds = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True) for t in y]
            predictions.extend(preds)
            actuals.extend(target)
            if batch_idx % 100 == 0:
                print(f'Completed{batch_idx}')

    final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
    final_df_to_csv("./predictions.csv")


if __name__ == "__main__":
    # train()
    evaluate()
