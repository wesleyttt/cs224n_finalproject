import torch
from transformers import RobertaTokenizer

class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, input_sequences, output_sequences):
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        input_sequence = self.input_sequences[index]
        output_sequence = self.output_sequences[index]

        encoded_input = self.tokenizer.encode_plus(input_sequence, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        encoded_output = self.tokenizer.encode_plus(output_sequence, padding='max_length', max_length=128, truncation=True, return_tensors='pt')

        input_ids = encoded_input['input_ids'].squeeze()
        output_ids = encoded_output['input_ids'].squeeze()


        return input_ids, output_ids
