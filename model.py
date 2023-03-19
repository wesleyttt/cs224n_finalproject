import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class Seq2SeqModel(nn.Module):
    def __init__(self, model, max_len=127):
        super(Seq2SeqModel, self).__init__()
        self.max_len = max_len
        self.batch_size = 32
        self.model = T5ForConditionalGeneration.from_pretrained(model)

    def forward(self, input_ids):
        encoder_output = self.encoder.generate(input_ids, max_length=input_ids.shape[1])
        decoder_output = self.decoder.generate(encoder_output,
                                               max_length=256,
                                               num_beams=5,
                                               no_repeat_ngram_size=2,
                                               early_stopping=True)
        return decoder_output
