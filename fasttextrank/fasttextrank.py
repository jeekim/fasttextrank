import networkx as nx
from fasttextrank.baserank import BaseRank
from fasttextrank.nlp import TextProcessor
from gensim.models import FastText
from itertools import combinations

import logging


class FastTextRank(BaseRank):
    """
    an implementation of a keyword extraction based on TextRank.

    this class inherits utility methods from BaseRank

    methods in this class used for extraction steps as follows.

    1. Process text: we used spacy for sentence splitting, lemmatization and POS tagging.

    2. Build a word graph: we used words belonging to the following POS categories (adjective, and noun).
    Using PageRank algorithm, we calculate scores of individual words.

    3. Select longest sequences of keywords: our assumption is that key phrases consist of combinations of nouns and
    adjectives. The system selects the longest sequences of nouns and adjectives within a sentence.

    4. Rank the longest sequences of keywords using individual scores of those keywords.

    5. Filter the candidates in their ranked order: we calculated a distance between key phrases using Word Mover’s
    Distance (WMD) [5] implemented in gensim and removed the low-ranked phrase if the distance is less than a threshold.

    """

    def __init__(self, model=None):
        """

        """
        super().__init__()
        self.graph = nx.Graph()
        self.model = FastText.load(model)

    def process_text(self, text):
        """
        Process text
        :param text:
        :return:
        """

        # calling spacy to process a text.
        doc = TextProcessor.process(text=text, max_length=10**5)

        self.sentences = doc.sentences

    def build_word_graph(self, window=None, pos=None):
        """
        Build a word graph
        :param window: window size for a max distance between words for adding their edge into a graph.
        :param pos: valid POS tags
        :return:
        """
        # use lemmas to build a word graph
        text = [(word, sentence.pos[i] in pos) for sentence in self.sentences for i, word in enumerate(sentence.lemmas)]

        # add nodes
        self.graph.add_nodes_from([word for word, valid in text if valid])

        # add edges
        for i, (node1, is_valid1) in enumerate(text):
            # skip if a word is neither a noun nor an adjective
            if not is_valid1:
                continue

            # add an edge with node2 within a window size
            for j in range(i + 1, min(i + window, len(text))):
                node2, is_valid2 = text[j]
                if is_valid2 and node1 != node2:
                    self.graph.add_edge(node1, node2)

    def select_longest_keyword_sequences(self, keywords):
        """
        Select longest sequences of keywords
        :param keywords: a list of keywords used for building a word graph
        :return:
        """
        for i, sentence in enumerate(self.sentences):
            # a list of valid token offsets
            sequence = []

            for j, token in enumerate(self.sentences[i].lemmas):
                # add an offset of a valid token into longest_sequence
                if token in keywords:
                    sequence.append(j)
                    if j < (sentence.length - 1):
                        continue
                # when a token is not valid check if there is a sequence
                if sequence:
                    start = sequence[0]
                    end = sequence[-1] + 1
                    self.add_candidate(words=sentence.words[start:end],
                                       lemmas=sentence.lemmas[start:end],
                                       pos=sentence.pos[start:end],
                                       sentence_id=i)
                # reset the list
                sequence = []

    def rank_candidates(self, normalized=False):
        """
        Rank the longest sequences
        :param normalized:
        :return:
        """
        # calculate PageRank score for each node using default values
        w = nx.pagerank_numpy(self.graph, alpha=0.85, weight=None)
        keywords = sorted(w, key=w.get, reverse=True)

        self.select_longest_keyword_sequences(keywords)

        for k in self.candidates.keys():
            tokens = self.candidates[k].lexical_form
            # calculate candidate score by summing individual scores
            self.weights[k] = sum([w[t] for t in tokens])
            if normalized:
                self.weights[k] /= len(tokens)

    def remove_duplicate_candidates(self, threshold=None):
        """
        Filter the candidates
        :return:
        """
        top_keys = sorted(self.weights, key=self.weights.get, reverse=True)
        keys_to_delete = []

        for (k1, k2) in combinations(top_keys, 2):
            if k1 == k2:
                continue
            # calculate distance between key phrases
            distance = self.model.wv.wmdistance(k1, k2)

            # remove a lower ranked candidate
            if distance < threshold:
                keys_to_delete.append(k2)

        # delete candidates selected from above
        for k in set(keys_to_delete):
            logging.warning(f'removing {k} from candidates.')
            del self.candidates[k]
            del self.weights[k]
