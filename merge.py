import openai as ai
import os
import sys

sys.path.insert(0, './wiki_info')
from wiki_api import request
from wiki_sentence_rank import most_similar
from wiki_util import separate_paragraph

sys.path.insert(0, './ner')
from ner import get_entities

"""
Set env variable OPENAI_API_KEY to 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.
"""

STANFORD_ORG_ID = "org-cB9pEd5E9DW746jSupwICGuh"

# ai.organization = STANFORD_ORG_ID
ai.apikey = os.getenv("OPENAI_API_KEY")


def get_model_info(model: str = "text-davinci-002") -> None:
    """
    Prints model info, including owner, id, parent, permissions
    :param model: id of model that you're querying
    :return: None
    """

    print(ai.Model.retrieve(model))


def get_wiki_info(lyric: str):
    """
    Retrieves relevant wiki information for named entities within a song lyric.
    :param lyric: lyric that we want to generate an annotation for
    :return: str
    """
    named_entities = get_entities(lyric)
    info = ""

    for ent in named_entities:
        requested = request(ent, lyric)
        split_p = separate_paragraph(requested)

        info += most_similar(split_p, lyric) + '.'
    
    return info


def get_t5_output(lyric: str):
    """
    Generates an annotation for a given song lyric utilizing T5. 
    :param lyric: lyric that we want to generate an annotation for
    :return: str
    """
    pass


def merge(lyric: str):
    """
    Merges the T5 annotation and wiki information into a unified annotation.
    :param lyric: lyric that we want to generate an annotation for
    :return: str
    """
    info = get_wiki_info(lyric)
    annotation = get_t5_output(lyric)
    input = f"Given an input \'{info}\', generate an output sentence merging in any relevant information from a second input \'{annotation}\'. You should treat the information in the second input as the truth, but the output should follow the same structure and formatting as the first input."

    merged = ai.Completion.create(
        engine="text-davinci-002",
        prompt=input,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7
    )

    output = merged.choices[0].text
    
    return output
    

def main():
    sample_lyric = "See both sides like Chanel, see on both sides like Chanel"
    print(merge(sample_lyric))


if __name__ == "__main__":
    main()
