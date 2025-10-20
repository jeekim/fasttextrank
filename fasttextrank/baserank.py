import abc
from collections import defaultdict

import logging


class Candidate(object):
    """
    data model for keyword/phrase candidates
    """
    def __init__(self):
        """
        variable initialization
        """
        self.lexical_form = []
        self.surface_forms = []
        self.sentence_ids = []
        self.pos_patterns = []


class BaseRank(metaclass=abc.ABCMeta):
    """
    an abstract class for concrete graph-based keyword extraction classes

    """
    def __init__(self):
        """
        variable initialisation
        """
        self.sentences = []
        self.candidates = defaultdict(Candidate)
        self.weights = {}

    def get_nbest(self, n=5):
        """
        get n best key phrases
        :param n:
        :return: a list of key phrases
        """
        best = sorted(self.weights, key=self.weights.get, reverse=True)
        nbest = [u.lower() for u in best[:min(n, len(best))]]

        if len(nbest) < n:
            logging.warning(f'{n} requested from {len(nbest)} candidates)')

        return nbest

    def add_candidate(self, words, lemmas, pos, sentence_id):
        """
        populate candidates for a document
        :param words:
        :param lemmas:
        :param pos:
        :param sentence_id:
        :return:
        """
        key = ' '.join(words).lower()

        self.candidates[key].lexical_form = lemmas
        self.candidates[key].surface_forms.append(words)
        self.candidates[key].pos_patterns.append(pos)
        self.candidates[key].sentence_ids.append(sentence_id)

    @abc.abstractmethod
    def process_text(self, text):
        pass

    @abc.abstractmethod
    def build_word_graph(self, window, pos):
        pass

    @abc.abstractmethod
    def rank_candidates(self, normalized):
        pass
