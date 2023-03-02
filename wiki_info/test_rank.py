from wiki_api import request
from wiki_sentence_rank import most_similar
from wiki_util import separate_paragraph

sample_lyric = "Yellow Ferrari like Pikachu"

named_entity = 'Pikachu'

requested = request(named_entity, sample_lyric)
split_p = separate_paragraph(requested)

print(requested)
print(most_similar(split_p, sample_lyric))
