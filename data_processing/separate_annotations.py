import json

# Create a JSON of song lyric and meaning

src = open('../Final Project Data/genius-expertise/annotation_info.json', 'r')
dest = open('./processed_annotations.json', 'w')

for line in src:
    data = json.loads(line)

    if 'edits_lst' in data:
        processed = {'song': data['song'],
                     'lyrics': data['lyrics'],
                     'annotation': data['edits_lst'][0]['content']}
    elif 'content' in data:
        processed = {'song': data['song'],
                     'lyrics': data['lyrics'],
                     'annotation': data['content']}
    else:
        continue

    dest.write(json.dumps(processed))
    dest.write('\n')
