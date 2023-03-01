import torch
import torch.nn as nn
import tqdm
from transformers import AutoModel

from cs224n_finalproject.model import Seq2SeqModel

# replace these with strings of our desired HF transformer
encoder = None
decoder = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = None
model_save_path = "model.params"

train_dataloader = None
test_dataloader = None

optimizer = None
criterion = None

def train():
    model = Seq2SeqModel(encoder, decoder)
    model = model.to(device)

    tokenizer = AutoModel.from_pretrained('bert-base-uncased')

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for batch_idx, (input_ids, attention_mask, output_ids) in enumerate(train_dataloader):
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
