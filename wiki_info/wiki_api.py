import requests
import json
from wiki_sentence_rank import most_similar

endpoint = "https://en.wikipedia.org/w/api.php"


def request(phrase: str, context: str) -> str:
    """
    Query a phrase to look up on the Wikipedia API

    :param phrase: phrase to search Wikipedia for
    :param context: if we need to disambiguate, we give the context to find
    the best one to use
    :return: the string that Wikipedia returns
    """

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": phrase,
        "exintro": 1,
        "explaintext": 1,
        "redirects": 1
    }

    response = requests.get(endpoint, params=params)

    data = json.loads(response.text)
    info = data['query']['pages']
    page_key = list(info.keys())[0]

    if 'missing' in info[page_key]:
        print('ERROR: could not find ', phrase, '.')
        return ""
    elif '(disambiguation)' in info[page_key]['title']:
        page = disambiguation(phrase, context)
    elif 'refer to' in info[page_key]['extract']:
        page = disambiguation(phrase, context)
    else:
        page = info[page_key]['extract']

    return page


def disambiguation(phrase: str, context: str) -> str:
    """
    If there's a phrase that contains multiple possibilities, we
    use the same sentence ranking algorithm to return the most similar or
    most likely output

    :param phrase: same search phrase
    :param context: context the phrase appeared in
    :return: the revised page
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": phrase,
        "explaintext": 1,
        "redirects": 1
    }

    response = requests.get(endpoint, params=params)

    data = json.loads(response.text)
    info = data['query']['pages']
    page_key = list(info.keys())[0]
    results = info[page_key]['extract']

    results_filtered = results.split("refer to")[1]

    # Take the list of other possibilities, put them in a list
    possibilities = []
    read_results = results_filtered.split('\n')

    for res in read_results:
        if '==' not in res and len(res) > 3:
            possibilities.append(res)

    # Using the same idea as the sentence ranking, rank the most likely output

    most_likely = most_similar(possibilities, context)
    most_likely_phrase = most_likely.split(', ')[0]
    page = request(most_likely_phrase, context)

    return page

