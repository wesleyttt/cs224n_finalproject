import torch
from transformers import BertTokenizer, BertModel
import numpy as np

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def sentence_rank(paragraph, target_sentence):
    # Tokenize the paragraph and target sentence
    tokenized_paragraph = tokenizer.encode_plus(paragraph, add_special_tokens=True, return_tensors='pt')
    tokenized_target_sentence = tokenizer.encode_plus(target_sentence, add_special_tokens=True, return_tensors='pt')

    # Get the BERT embeddings for each sentence in the paragraph
    with torch.no_grad():
        bert_embeddings = model(tokenized_paragraph['input_ids'], tokenized_paragraph['attention_mask'])[0]

    bert_embeddings = bert_embeddings[:, 0, :]

    # Get the BERT embeddings for the target sentence
    with torch.no_grad():
        target_sentence_embeddings = model(tokenized_target_sentence['input_ids'],
                                           tokenized_target_sentence['attention_mask'])[0]

    # Calculate the cosine similarity between the target sentence and each sentence in the paragraph
    similarities = []
    for i in range(bert_embeddings.shape[0]):
        numerator = np.dot(target_sentence_embeddings[0].numpy(), bert_embeddings[i].numpy())
        denominator = (np.linalg.norm(target_sentence_embeddings[0].numpy()) * np.linalg.norm(bert_embeddings[i].numpy()))
        cos_sim = numerator / denominator
        similarities.append(cos_sim)

    # Return the maximum similarity as the ranking score for the target sentence
    return similarities[0]


paragraph = "Apples are a really yummy fruit. Apple Inc. is an American multinational technology company headquartered in Cupertino, California, United States. Apple is the largest technology company by revenue (totaling US$365.8 billion in 2021) and, as of June 2022, is the world's biggest company by market capitalization, the fourth-largest personal computer vendor by unit sales and second-largest mobile phone manufacturer. It is one of the Big Five American information technology companies, alongside Alphabet (Google), Amazon, Meta (Facebook), and Microsoft."

target_sentence = "I like eating apples"

score = sentence_rank(paragraph, target_sentence)

print(f"Score: {score}")