import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoModelForCausalLM

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, max_len=512):
        super(Seq2SeqModel, self).__init__()
        self.max_len = max_len
        self.tokenizer = AutoModel.from_pretrained('bert-base-uncased')
        self.encoder = AutoModel.from_pretrained(encoder)
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder)
        self.linear = nn.Linear(self.decoder.config.hidden_size, self.tokenizer.vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)[0]
        decoder_output = self.decoder(decoder_input_ids, encoder_output)
        logits = self.linear(decoder_output)
        return logits