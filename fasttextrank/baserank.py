import abc
from collections import defaultdict
import logging
from typing import List, Dict, DefaultDict, Set, Any # Any for graph type in abstract method if needed
from .type import Sentence # Assuming type.py is in the same directory

class Candidate(object):
    """
    Data model for keyword/phrase candidates.

    Stores information about a potential keyphrase, including its various forms,
    grammatical structure, and locations within the source text.
    """
    def __init__(self) -> None:
        """
        Initializes a Candidate object with empty attributes.
        
        Attributes:
            lexical_form (List[str]): A list of lemmas representing the canonical form of the candidate.
            surface_forms (List[List[str]]): A list of lists, where each inner list contains the surface words
                                  of a specific occurrence of the candidate.
            sentence_ids (List[int]): A list of integer IDs for sentences where the candidate appears.
            pos_patterns (List[List[str]]): A list of lists, where each inner list contains the Part-of-Speech tags
                                 for a specific occurrence of the candidate.
        """
        self.lexical_form: List[str] = []  # Canonical form (lemmas)
        self.surface_forms: List[List[str]] = [] # Observed forms (words)
        self.sentence_ids: List[int] = []  # IDs of sentences containing the candidate
        self.pos_patterns: List[List[str]] = []  # POS tag sequences for each surface form


class BaseRank(metaclass=abc.ABCMeta):
    """
    An abstract base class for graph-based keyword extraction algorithms.

    This class defines the common interface and shared functionalities for specific
    keyword extraction implementations (e.g., TextRank, FastTextRank). It manages
    sentences, candidates, and their weights.
    """
    def __init__(self) -> None:
        """
        Initializes BaseRank attributes.

        Attributes:
            sentences (List[Sentence]): A list to store processed Sentence objects from the input text.
            candidates (DefaultDict[str, Candidate]): A dictionary-like object storing Candidate objects,
                                      keyed by their lowercase string representation.
            weights (Dict[str, float]): A dictionary to store the calculated ranking scores (weights) for candidates.
            graph (Any): Placeholder for a graph object, typically networkx.Graph in subclasses.
        """
        self.sentences: List[Sentence] = []
        self.candidates: DefaultDict[str, Candidate] = defaultdict(Candidate) # Stores Candidate objects
        self.weights: Dict[str, float] = {} # Stores weights/scores for candidates
        self.graph: Any = None # Concrete subclasses will define the graph type, e.g., nx.Graph

    def get_nbest(self, n: int = 5) -> List[str]:
        """
        Retrieves the top N highest-ranked keyphrases.

        :param n: int, the number of top keyphrases to return. Defaults to 5.
        :return: list[str], a list of the top N keyphrases (as lowercase strings).
        """
        # Sort candidates by their weights in descending order
        best: List[str] = sorted(self.weights, key=lambda k: self.weights[k], reverse=True)
        # Get the top N, or fewer if not enough candidates exist
        nbest: List[str] = [u.lower() for u in best[:min(n, len(best))]]

        if len(nbest) < n:
            logging.info(f'{n} keyphrases requested, but only {len(nbest)} candidates were available/ranked.')

        return nbest

    def add_candidate(self, words: List[str], lemmas: List[str], pos: List[str], sentence_id: int) -> None:
        """
        Adds or updates a candidate keyphrase in the `self.candidates` collection.

        The candidate is identified by a key generated from the lowercase version of its words.
        This method populates the attributes of the Candidate object associated with this key.

        :param words: list[str], the surface form (actual words) of the candidate occurrence.
        :param lemmas: list[str], the lemmatized form of the candidate occurrence.
        :param pos: list[str], the Part-of-Speech tags for the candidate occurrence.
        :param sentence_id: int, the ID of the sentence where this candidate occurrence was found.
        :return: None
        """
        # Create a unique key for the candidate based on its surface form (words)
        key: str = ' '.join(words).lower()

        # Populate the Candidate object's attributes
        self.candidates[key].lexical_form = lemmas # Store the canonical lemmatized form
        self.candidates[key].surface_forms.append(words) # Add this specific surface form
        self.candidates[key].pos_patterns.append(pos) # Add the POS pattern for this surface form
        self.candidates[key].sentence_ids.append(sentence_id) # Record the sentence ID

    @abc.abstractmethod
    def process_text(self, text: str) -> None:
        """
        Abstract method for processing the input text.

        Subclasses must implement this to parse the text, typically involving
        sentence splitting, tokenization, lemmatization, and POS tagging,
        and populate `self.sentences`.

        :param text: str, the raw input text to process.
        """
        pass

    @abc.abstractmethod
    def build_word_graph(self, window: int, pos: Set[str]) -> None:
        """
        Abstract method for building the word graph.

        Subclasses must implement this to construct a graph from the processed
        sentences, where nodes are words and edges represent co-occurrence or
        other relationships. The graph should typically be stored in `self.graph`.

        :param window: int, the co-occurrence window size.
        :param pos: set[str], a set of valid Part-of-Speech tags to consider for graph nodes.
        """
        pass

    @abc.abstractmethod
    def rank_candidates(self, normalized: bool) -> None:
        """
        Abstract method for ranking candidates.

        Subclasses must implement this to calculate scores for the identified
        candidates, typically using graph-based ranking algorithms (like PageRank)
        on the word graph, and then deriving candidate scores. Results should
        be stored in `self.weights`.

        :param normalized: bool, whether to normalize candidate scores.
        """
        pass
