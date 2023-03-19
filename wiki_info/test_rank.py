from wiki_api import request
from wiki_sentence_rank import most_similar
from wiki_util import separate_paragraph

import sys
sys.path.insert(0, '../ner')

from ner import get_entities

sample_lyric = "Yellow Ferrari like Pikachu. See both sides like Chanel, see on both sides like Chanel"

named_entities = get_entities(sample_lyric)

for ent in named_entities:
    requested = request(ent, sample_lyric)
    split_p = separate_paragraph(requested)

    print(requested)
    print(most_similar(split_p, sample_lyric))
