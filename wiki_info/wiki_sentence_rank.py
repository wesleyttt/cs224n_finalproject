import torch
from transformers import BertTokenizer, BertModel

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def most_similar(sentences: list, target_sentence: str) -> str:
    """
        Given a target sentence, we want to rank the sentences of a paragraph in similarity to
        the target sentence.

        :param sentences: list of sentences that we want to find most similar sentence to our target
        :param target_sentence: target sentence we are quering
        :return: List of cosine similarity score, higher is better
        """
    # Tokenize the sentences and target sentence
    tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    tokenized_target = tokenizer(target_sentence, padding=True, truncation=True, return_tensors='pt')

    # Encode the sentences and target sentence using BERT
    with torch.no_grad():
        sentence_embeddings = model(**tokenized_sentences)[0][:, 0, :]
        target_embedding = model(**tokenized_target)[0][:, 0, :]

    # Calculate the cosine similarity between the target sentence and each sentence in the list
    cosine_similarities = torch.nn.functional.cosine_similarity(target_embedding, sentence_embeddings)

    # Find the index of the most similar sentence
    most_similar_index = cosine_similarities.argmax().item()

    return sentences[most_similar_index]
