import requests
import json


def request(phrase: str) -> str:
    """
    Query a phrase to look up on the Wikipedia API

    :param phrase: phrase to search Wikipedia for
    :return: the string that Wikipedia returns
    """
    endpoint = "https://en.wikipedia.org/w/api.php"

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
        page = 'ERROR: could not find ' + phrase + '.'
    elif '(disambiguation)' in info[page_key]['title']:
        page = 'ERROR: be more specific with what you are looking for'
    else:
        page = info[page_key]['extract']

    return page


if __name__ == '__main__':
    print(request("Bengals"))
