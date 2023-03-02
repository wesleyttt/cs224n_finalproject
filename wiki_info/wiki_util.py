def separate_paragraph(paragraph: str) -> list:
    """
    Given a paragraph, separate it into a list of sentences

    :param paragraph: long string representing the paragraph
    :return: list of each individual sentence
    """
    new_paragraph = paragraph.split('. ')

    return list(new_paragraph)
