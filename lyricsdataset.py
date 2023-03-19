import torch
from transformers import AutoTokenizer

class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, input_sequences, output_sequences):
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        input_sequence = "Summarize: " + self.input_sequences[index]
        output_sequence = self.output_sequences[index]

        source = self.tokenizer.batch_encode_plus([input_sequence], max_length=256, pad_to_max_length=True,
                                                  return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([output_sequence], max_length=256, pad_to_max_length=True,
                                                  return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }
