import requests
from typing import List, Dict
from requests.adapters import HTTPAdapter, Retry
import random
import json

#MY CODE#################################################################
import gensim
import joblib
import re
import time
import string
import spacy
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")
nlp.max_length = np.inf
stop_words = nlp.Defaults.stop_words

RE_URL = re.compile(r'\S*https?:\S*')
RE_NON_PRINTABLE = re.compile(r'[\x00-\x1F\x7F]')
RE_NUMBERS = re.compile(r'\d+')
RE_SPACES = re.compile(r'\s+')


def process_text(text):
    text = RE_URL.sub('', text) # Remove URLs
    text = text.lower() # Lowercase
    doc = nlp(text)
    lemmas = [word.lemma_ for word in doc] # Lemmatize
    lemmas = [RE_NON_PRINTABLE.sub('', lemma) for lemma in lemmas] # Remove non-printable characters
    lemmas = [RE_NUMBERS.sub('', lemma) for lemma in lemmas] # Remove numbers
    lemmas = [lemma for lemma in lemmas if lemma not in stop_words] # Remove stop words
    text = " ".join(lemmas)
    table = str.maketrans('', '', string.punctuation) # Remove punctuation
    text = text.translate(table)
    text = text.replace('\n', ' ')  # Remove new line
    text = text.strip()
    text = RE_SPACES.sub(' ', text) # Remove extra spaces
    return text

def update_user_text(username, new_text, filename='users.json'):
    # Load the existing data
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    # Update the user's text
    if username in data:
        data[username] += ' ' + new_text
    else:
        data[username] = new_text

    # Save the updated data
    with open(filename, 'w') as f:
        json.dump(data, f)

def get_user_text(username, filename='users.json'):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    return data.get(username, None)

        
######################################
        
ENDPOINT_GET_WRITINGS = "https://erisk.irlab.org/{}/getwritings/{}"
ENDPOINT_SUBMIT_DECISIONS = "https://erisk.irlab.org/{}/submit/{}/{}"
USERS_PATH = "users.txt"


def _save_users(users: List[str]):
    with open(USERS_PATH, 'w') as fp:
        for nick in users:
            fp.write("%s\n" % nick)


def _load_users():
    with open(USERS_PATH, 'r') as fp:
        return set([nick.strip() for nick in fp.readlines()])


class DummyClient:
    def __init__(self, service: str, token: str, number_of_runs: int):
        self.service = service
        self.token = token
        self.number_of_runs = number_of_runs
        self.current_sequence = 0
        self.users = set()
        self.last_decisions = {}
        self.last_decisions2 = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.doc2vec = gensim.models.Doc2Vec.load("TASK 2/modelos/doc2vec_model")
        self.transformer = SentenceTransformer("facebook/bart-base").to(self.device)
        self.ens_doc = joblib.load("TASK 2/modelos/ensemble_doc2vec.joblib") #doc2vec good
        self.ens_trans = joblib.load("TASK 2/modelos/ensemble_transformer.joblib") #transformer less good

    def get_writings(self, retries: int, backoff: float) -> Dict:
        session = requests.Session()
        retries = Retry(total=retries,
                        backoff_factor=backoff,
                        status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(
            ENDPOINT_GET_WRITINGS.format(self.service, self.token))
        return json.loads(response.text)

    def submit_decission(self, round, writings: List[Dict], retries, backoff):
        decisions = []
        decisions2 = []
        current_round_active_users = set()
        for writing in writings:
            decision = {}
            decision2 = {}
            decision["nick"] = writing["nick"]
            decision2["nick"] = writing["nick"]
            start = time.time()
            update_user_text(writing["nick"], process_text(writing['content']))
            sentence = get_user_text(writing["nick"])
            print(time.time()-start)
            #decision1
            start = time.time()
            vector = self.doc2vec.infer_vector(sentence.split())
            decision["decision"] = str(int(self.ens_doc.predict([vector]).squeeze()))
            decision["score"] = np.round(max(self.ens_doc.predict_proba([vector]).squeeze()),3)
            print(time.time()-start)
            #decision2
            start = time.time()
            embeddings = self.transformer.encode(sentence).squeeze()
            decision2["decision"] = str(int(self.ens_trans.predict([embeddings]).squeeze()))
            decision2["score"] = np.round(max(self.ens_trans.predict_proba([embeddings]).squeeze()),3)
            print(time.time()-start)
            ############################################
            decisions.append(decision)
            decisions2.append(decision2)

            self.last_decisions[writing["nick"]] = decision
            self.last_decisions2[writing["nick"]] = decision2

            current_round_active_users.add(writing["nick"])

        # We now add the users thar finished the writings in previous rounds
        for non_active_user in self.users - current_round_active_users:
            decisions.append(self.last_decisions[non_active_user] if non_active_user in self.last_decisions else {
                             "nick": non_active_user, "decision": random.choice([0, 1]), "score": random.random()})
            decisions2.append(self.last_decisions2[non_active_user] if non_active_user in self.last_decisions2 else {
                    "nick": non_active_user, "decision": random.choice([0, 1]), "score": random.random()})

            
        session = requests.Session()
        retries = Retry(total=retries,
                        backoff_factor=backoff,
                        status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        for run in range(self.number_of_runs):
            if run == 0:
                response = session.post(ENDPOINT_SUBMIT_DECISIONS.format(
                    self.service, self.token, run), json=decisions)
            if run == 1:
                response = session.post(ENDPOINT_SUBMIT_DECISIONS.format(
                    self.service, self.token, run), json=decisions2)

            if response.status_code != 200:
                print(response.text)
            else:
                print("Round {}: submmited decissions {} stored decissions {} for run {}".format(round, len(decisions),
                                                                                                 len(json.loads(response.text)), run))

    def run(self, retries: int, backoff: float):
        writings = self.get_writings(retries, backoff)
        if len(writings) == 0:
            print("All rounds processed")
            return
        if writings[0]["number"] == 0:
            # it is the first round
            self.users = set([writing["nick"] for writing in writings])
            _save_users(self.users)
        else:
            self.users = _load_users()
        
        while len(writings) > 0:
            print(len(writings))
            self.submit_decission(
                writings[0]["number"], writings, retries, backoff)
            writings = self.get_writings(retries, backoff)
        print("All rounds processed")


def main() -> int:
    client = DummyClient("",
                         "", 2)
    client.run(5, 0.1)


if __name__ == '__main__':
    main()
