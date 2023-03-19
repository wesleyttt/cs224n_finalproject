import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from lyricsdataset import LyricsDataset
from data_processing.read_data import read_data
from sklearn.model_selection import train_test_split

lyrics, annotations = read_data()
lyrics_train, lyrics_test, annotations_train, annotations_test = train_test_split(lyrics, annotations, test_size=0.2)
dataset = LyricsDataset(lyrics_test, annotations_test)
LyricsLoader = torch.utils.data.DataLoader(dataset, batch_size=1)

tokenizer = AutoTokenizer.from_pretrained("t5-small")

model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.load_state_dict(torch.load("model.params"))


input_ids = tokenizer.encode("summarize: My Louboutins new, so my bottoms, they is redder", max_length=512, return_tensors="pt")
#print(tokenizer.decode(_['input_ids']))
#print("Lyrics_test[200]: ", lyrics_test[200])
#print("Tokenizer decoding: ", tokenizer.decode(i["input_ids"], skip_special_tokens=True))
output = model.generate(input_ids,
                        min_length=24,
                        max_length=512,
                        num_beams=6,
                        no_repeat_ngram_size=1,
                        early_stopping=False,
                        temperature=0.7,
                        remove_invalid_values=True)

print("Generation: ", tokenizer.decode(output[0], skip_special_tokens=True))
