import json

"""
When we put the code into the VM, we need to put the processed_annotations.json
file in the data_processing folder
"""
PATH = 'data_processing/processed_annotations.json'


def read_data() -> tuple[list[str], list[str]]:
    """
    Read the JSON file containing the lyrics and annotations, and
    populate a list to feed into the model

    :param: none
    :rtype: tuple with two lists inside it
    :return: (lyrics, annotation)
    lyrics is a list containing the lyrics
    annotation is a list containing the annotations of a lyric
    the ith'
    """

    f = open(PATH, 'r')

    lyrics = []
    annotations = []
    numlines = 0

    for line in f:
        numlines += 1
        data = json.loads(line)
        lyrics.append(data['lyrics'])
        annotations.append(data['annotation'])

    return lyrics, annotations


if __name__ == '__main__':
    read_data()