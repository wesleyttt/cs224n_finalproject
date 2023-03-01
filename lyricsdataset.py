import torch
import torch.nn as nn
from transformers import AutoConfig

class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, input_sequences, output_sequences):
        # TODO: integrate our genius dataset and process the data
        self.input_sequences = None
        self.output_sequences = None
        self.tokenizer = AutoConfig.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        input_sequence = self.input_sequences[index]
        output_sequence = self.output_sequences[index]

        encoded_input = self.tokenizer.encode_plus(input_sequence, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        encoded_output = self.tokenizer.encode_plus(output_sequence, padding='max_length', max_length=128, truncation=True, return_tensors='pt')

        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        output_ids = encoded_output['input_ids']

        return input_ids.squeeze(), attention_mask.squeeze(), output_ids.squeeze()
