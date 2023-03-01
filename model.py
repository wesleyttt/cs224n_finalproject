import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, AutoConfig, RobertaForCausalLM

class Seq2SeqModel(nn.Module):
    def __init__(self, model, max_len=127):
        super(Seq2SeqModel, self).__init__()
        self.max_len = max_len
        self.batch_size = 32
        self.encoder = RobertaModel.from_pretrained(model)
        config = AutoConfig.from_pretrained("roberta-base")
        config.is_decoder = True
        config.add_cross_attention = True
        self.decoder = RobertaForCausalLM.from_pretrained("roberta-base", config=config)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def forward(self, input_ids, decoder_input_ids):
        encoder_output = self.encoder(input_ids)[0]
        decoder_output = self.decoder(decoder_input_ids, encoder_hidden_states=encoder_output)
        return decoder_output
