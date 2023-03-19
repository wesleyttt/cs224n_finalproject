import stanza

stanza.download('en')  # download English model
nlp = stanza.Pipeline('en')  # initialize English neural pipeline


def get_entities(lyric: str) -> list:
    """
    Given a song lyric, extract the song lyrics from it

    :param lyric: The string representing the lyric
    :return: A list of named entities
    """
    doc = nlp(lyric)
    ents = set()
    words = set()

    for sentence in doc.sentences:
        for span in sentence.ents:
            ents.add(span.text)

    return list(ents)


get_entities("Yellow Ferrari Nicki Minaj like Pikachu")

