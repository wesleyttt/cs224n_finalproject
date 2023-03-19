import openai as ai
import os
import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from lyricsdataset import LyricsDataset
from data_processing.read_data import read_data
from sklearn.model_selection import train_test_split

sys.path.insert(0, './wiki_info')
from wiki_api import request
from wiki_sentence_rank import most_similar
from wiki_util import separate_paragraph

sys.path.insert(0, './ner')
from ner import get_entities, get_entities_finetune

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


def get_wiki_info_finetune(lyric: str):
    """
    Retrieves relevant wiki information for named entities within a song lyric.
    :param lyric: lyric that we want to generate an annotation for
    :return: str
    """
    named_entities = get_entities_finetune(lyric)
    info = ""

    for ent in named_entities:
        requested = request(ent, lyric)
        split_p = separate_paragraph(requested)

        info += most_similar(split_p, lyric) + '.'

    return info


def get_t5_output(lyric: str, tokenizer):
    """
    Generates an annotation for a given song lyric utilizing T5.
    :param lyric: lyric that we want to generate an annotation for
    :return: str
    """
    # tokenizer = AutoTokenizer.from_pretrained("t5-small")

    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.load_state_dict(torch.load("model.params"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input_ids = tokenizer.encode(f"summarize: {lyric}", max_length=512, return_tensors="pt").to(device)

    output = model.generate(input_ids,
                            min_length=24,
                            max_length=512,
                            num_beams=6,
                            no_repeat_ngram_size=1,
                            early_stopping=False,
                            temperature=0.7,
                            remove_invalid_values=True)

    return tokenizer.decode(output[0], skip_special_tokens=True)


def merge(lyric: str, tokenizer):
    """
    Merges the T5 annotation and wiki information into a unified annotation.
    :param lyric: lyric that we want to generate an annotation for
    :return: str
    """
    info = get_wiki_info(lyric)
    annotation = get_t5_output(lyric, tokenizer)
    input = f"Given an input \'{annotation}\', generate an output sentence merging in any relevant information from a second input \'{info}\'. You should treat the information in the second input as the truth, but the output should follow the same structure and formatting as the first input. As an example, suppose this is the input: Louboutins is a brand of luxury watches. The most expensive designer shoes are the highest quality, and Lil Uzi Vert's shoes are redder because they are new. Suppose the following is the information: Christian Louboutin is a high-end French fashion house that specializes in shoes. You would merge them together to get this output: Christian Louboutin is a high-end French fashion house. Louboutin are most famous for their footwear, distinguishable by the trademark red sole. With continuous use, the redness fades over time. Lil Uzi Vert's Louboutin shoes are new, meaning that the sole of his shoes are at their reddest."
    # input = f"Given an input \'{annotation}\', generate an output sentence merging in any relevant information from a second input \'{info}\'. You should treat the information in the second input as the truth, but the output should follow the same structure and formatting as the first input."

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


def merge_finetune(lyric: str, tokenizer):
    """
    Merges the T5 annotation and wiki information into a unified annotation.
    :param lyric: lyric that we want to generate an annotation for
    :return: str
    """
    info = get_wiki_info_finetune(lyric)
    annotation = get_t5_output(lyric, tokenizer)
    input = f"Given an input \'{annotation}\', generate an output sentence merging in any relevant information from a second input \'{info}\'. You should treat the information in the second input as the truth, but the output should follow the same structure and formatting as the first input. As an example, suppose this is the input: Louboutins is a brand of luxury watches. The most expensive designer shoes are the highest quality, and Lil Uzi Vert's shoes are redder because they are new. Suppose the following is the information: Christian Louboutin is a high-end French fashion house that specializes in shoes. You would merge them together to get this output: Christian Louboutin is a high-end French fashion house. Louboutin are most famous for their footwear, distinguishable by the trademark red sole. With continuous use, the redness fades over time. Lil Uzi Vert's Louboutin shoes are new, meaning that the sole of his shoes are at their reddest."
    # input = f"Given an input \'{annotation}\', generate an output sentence merging in any relevant information from a second input \'{info}\'. You should treat the information in the second input as the truth, but the output should follow the same structure and formatting as the first input."

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