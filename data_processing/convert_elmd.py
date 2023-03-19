import json
import os

"""
The original dataset from elmd2 is stored in the following format:
[json, json, ...]
Each json is a sentence, formatted as:
{
    "text": ...,
    "entities": list of entities
}
We need to tokenize the text, and convert it to this format
[[json, ...], [json, ...], ...]
where each json is formatted as:
{
    "text": corresponding word,
    "ner": B if it is an entity and first word of sentence, O if it isn't an entity, E otherwise
}

"""

def convert_sentence(sent):
    """
    Input: sentence in JSON format
    Output: list of JSON representing words in a sentence
    """
    converted = []
    toks = sent['text'].split()
    ents = [sent['text'][e['startChar']:e['endChar']] for e in sent['entities']]

    first = True
    for tok in toks:
        if tok in ents:
            if first:
                processed = {'text': tok,
                             'ner': "B"}
            else:
                processed = {'text': tok,
                             'ner': "E"}
        else:
            processed = {'text': tok,
                         'ner': "O"}
        converted.append(processed)

    return converted


train = open('../data/ner/en_musicner.train.json', 'w')
dev = open('../data/ner/en_musicner.dev.json', 'w')
test = open('../data/ner/en_musicner.test.json', 'w')

train_res = []
dev_res = []
test_res = []
count = 0
for file in os.listdir('../data/elmd2'):
    count += 1
    if file.endswith('.json'):
        with open(os.path.join('../data/elmd2', file), 'r') as f:
            data = json.load(f)

            for sentence in data:
                res = convert_sentence(sentence)

            if count <= 10408:
                train_res.append(res)
            elif count <= 10408 + 1301:
                dev_res.append(res)
            else:
                test_res.append(res)

train.write(json.dumps(train_res))
test.write(json.dumps(test_res))
dev.write(json.dumps(dev_res))

