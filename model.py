import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder_model_name, decoder_model_name, output_size, device):
        super(Seq2SeqModel, self).__init__()

        self.encoder_model = AutoModel.from_pretrained(encoder_model_name)
        self.encoder_hidden_size = self.encoder_model.config.hidden_size

        self.decoder_model = AutoModel.from_pretrained(decoder_model_name)
        self.decoder_hidden_size = self.decoder_model.config.hidden_size

        self.decoder_to_output = nn.Linear(self.decoder_hidden_size, output_size)

        self.device = device

    def forward(self, input_seq, target_seq=None, teacher_forcing_ratio=0.5):
        input_seq = input_seq.to(self.device)
        if target_seq is not None:
            target_seq = target_seq.to(self.device)

        # Encode the input sequence
        encoder_output, encoder_hidden = self.encoder_model(input_seq)

        # Initialize the decoder hidden state with the encoder's final hidden state
        decoder_hidden = encoder_hidden

        output_seq = []
        for i in range(target_seq.shape[1]):

            # Use teacher forcing to decide whether to use the true target output or the predicted output
            use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False

            if use_teacher_forcing and target_seq is not None:
                decoder_input = target_seq[:, i, :]
            else:
                decoder_input = self.decoder_to_output(decoder_hidden.last_hidden_state)

            # Pass the decoder input through the decoder
            decoder_output = self.decoder_model(decoder_input, decoder_hidden)
            decoder_hidden = decoder_output

            # Append the decoder output to the output sequence
            output_seq.append(decoder_output.last_hidden_state.unsqueeze(1))

        output_seq = torch.cat(output_seq, dim=1)
        return output_seq