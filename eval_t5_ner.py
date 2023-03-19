from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from lyricsdataset import LyricsDataset
from data_processing.read_data import read_data
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from ignite.metrics import Rouge
from merge import merge

import math
import re
from collections import Counter

WORD = re.compile(r"\w+")
ITERATIONS = 70000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained('t5-small')

m = Rouge(variants=["L", 1], multiref="best")

# Data
lyrics, annotations = read_data()
lyrics_train, lyrics_test, annotations_train, annotations_test = train_test_split(lyrics, annotations, test_size=0.2)

references = []
hypotheses = []

# Load pre-trained model (weights)
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model = model.to(device)

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

annotation_cos_sim = 0
lyrics_cos_sim = 0
cos = nn.CosineSimilarity(dim=0)

for i in tqdm(range(len(annotations_test))):
    references.append([annotations_test[i].split()])

    text = lyrics_test[i]
    output = merge(text, tokenizer)
    # # Encode a text inputs
    # text = "Summarize: " + lyrics_test[i]
    # indexed_tokens = tokenizer.encode(text)

    # # Encode the lyric using the tokenizer
    # input_ids = tokenizer.encode(text, max_length=256, return_tensors='pt', pad_to_max_length=True).to(device)
    # output = model.generate(input_ids=input_ids, num_beam_groups=2, max_length=256, num_beams=6, early_stopping=True,
    #                                 no_repeat_ngram_size=1)

    # # Decode the output text
    # output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    hypotheses.append(output.split())

    m.update(([output.split()], [annotations_test[i].split()]))

    annotation_vec = text_to_vector(annotations_test[i])
    lyrics_vec = text_to_vector(lyrics_test[i])
    output_vec = text_to_vector(output)

    annotation_cos_sim += get_cosine(annotation_vec, output_vec)
    lyrics_cos_sim += get_cosine(lyrics_vec, output_vec)

    if i == ITERATIONS:
        break

# Calculate BLEU
bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
print(bleu)

# calculate ROUGE (note: I used pytorch rouge API)
print("Rouge: ", m.compute())

print("Annotations: ", annotation_cos_sim, "Lyrics: ", lyrics_cos_sim)

annotation_cos_sim /= min(len(annotations_test), ITERATIONS)
lyrics_cos_sim /= min(len(annotations_test), ITERATIONS)

print("Custom metric: ", (0.5 * (annotation_cos_sim - lyrics_cos_sim)))

