import networkx as nx
from .baserank import BaseRank, Candidate # Added Candidate here
from .nlp import TextProcessor
from .type import Sentence # Added Sentence here
from gensim.models import FastText
from itertools import combinations
import logging
from typing import List, Dict, Tuple, Optional, Callable, Any, Set


class GraphHandler:
    """
    Handles the creation and processing of the word graph.

    This class is responsible for building a graph from sentences and calculating
    node rankings (e.g., PageRank) on that graph.
    """
    def __init__(self) -> None:
        """Initializes the GraphHandler with an empty NetworkX graph."""
        self.graph: nx.Graph = nx.Graph()

    def build_word_graph(self, sentences: List[Sentence], window: Optional[int] = 4, pos: Optional[Set[str]] = None) -> nx.Graph:
        """
        Builds a word graph from a list of processed sentences.

        Nodes in the graph are lemmas of words that match the specified Part-of-Speech tags.
        Edges are added between lemmas that co-occur within a specified window in the sentences.

        :param sentences: list[Sentence], a list of Sentence objects (from .type).
        :param window: int, the maximum distance (in tokens) between words for an edge to be added.
        :param pos: set[str], a set of valid Part-of-Speech tags for words to be included as nodes.
        :return: networkx.Graph, the constructed word graph.
        """
        if pos is None: # Default POS tags if not provided
            pos = {'NOUN', 'ADJ'}
        if window is None: # Default window if not provided
            window = 4

        # Extract lemmas and a validity flag (based on POS) for each token in each sentence
        text: List[Tuple[str, bool]] = []
        for sentence in sentences:
            for i, word_lemma in enumerate(sentence.lemmas):
                text.append((word_lemma, sentence.pos[i] in pos))

        # Clear any existing graph structure for fresh processing
        self.graph.clear()
        
        # Add valid lemmas as nodes to the graph
        self.graph.add_nodes_from([word for word, valid in text if valid])

        # Add edges based on co-occurrence within the specified window
        for i, (node1, is_valid1) in enumerate(text):
            if not is_valid1: # Skip if the token is not a valid POS type
                continue
            # Iterate over subsequent tokens within the window
            for j in range(i + 1, min(i + window, len(text))):
                node2, is_valid2 = text[j]
                if is_valid2 and node1 != node2: # Add edge if valid and different
                    self.graph.add_edge(node1, node2)
        return self.graph

    def get_graph(self) -> nx.Graph:
        """
        Returns the current word graph.

        :return: networkx.Graph, the internal graph object.
        """
        return self.graph

    def calculate_pagerank(self, alpha: float = 0.85, weight: Optional[str] = None) -> Dict[str, float]:
        """
        Calculates PageRank scores for the nodes in the graph.

        :param alpha: float, damping parameter for PageRank. Default is 0.85.
        :param weight: str or None, the edge data key to use as weight. If None, edges are unweighted.
        :return: dict, a dictionary mapping each node (word lemma) to its PageRank score.
                 Returns an empty dictionary if the graph is not built or is empty.
        """
        if not self.graph or self.graph.number_of_nodes() == 0: # Check if graph is empty or not built
            logging.warning("Graph is not built or is empty. Returning empty PageRank scores.")
            return {}
        return nx.pagerank(self.graph, alpha=alpha, weight=weight)


# Define a specific type for the add_candidate_func callback
AddCandidateCallable = Callable[[List[str], List[str], List[str], int], None]

class CandidateExtractor:
    """
    Extracts candidate keyphrases from processed text.

    This class identifies sequences of words (based on ranked keywords) in sentences
    that could be potential keyphrases.
    """
    def __init__(self, add_candidate_func: AddCandidateCallable) -> None:
        """
        Initializes the CandidateExtractor.

        :param add_candidate_func: function, a callback function (e.g., `BaseRank.add_candidate`)
                                   used to register extracted candidates.
        """
        self.add_candidate: AddCandidateCallable = add_candidate_func

    def extract_sequences(self, sentences: List[Sentence], keywords: List[str]) -> None:
        """
        Selects the longest sequences of keywords appearing adjacently in sentences.

        These sequences are then registered as candidates using the `add_candidate_func`.

        :param sentences: list[Sentence], a list of Sentence objects from .type.
        :param keywords: list[str], a list of ranked keywords (lemmas) to look for.
        :return: None (candidates are added via the callback function).
        """
        for i, sentence in enumerate(sentences):
            sequence: List[int] = [] # Holds indices of current adjacent keyword sequence
            for j, token in enumerate(sentence.lemmas):
                if token in keywords:
                    sequence.append(j)
                    # If it's not the last token, continue to check for longer sequence
                    if j < (sentence.length - 1):
                        continue
                
                # If a sequence was formed and the current token breaks it (or end of sentence)
                if sequence:
                    start_idx: int = sequence[0]
                    end_idx: int = sequence[-1] + 1 # Slice goes up to, but not including, end_idx
                    # Register the identified sequence as a candidate
                    self.add_candidate(words=sentence.words[start_idx:end_idx],
                                       lemmas=sentence.lemmas[start_idx:end_idx],
                                       pos=sentence.pos[start_idx:end_idx],
                                       sentence_id=i)
                sequence = [] # Reset for the next potential sequence


class CandidateRanker:
    """
    Ranks candidate keyphrases.

    This class calculates scores for candidates based on the scores of their
    constituent words (e.g., derived from PageRank).
    """
    def rank(self, candidates: Dict[str, Candidate], word_scores: Dict[str, float], normalized: bool = False) -> Dict[str, float]:
        """
        Ranks candidates based on the sum of their constituent word scores.

        :param candidates: dict, a dictionary of Candidate objects (from BaseRank),
                           keyed by their string representation.
        :param word_scores: dict, a dictionary mapping word lemmas to their scores.
        :param normalized: bool, if True, normalizes the candidate's score by its length (number of tokens).
        :return: dict, a dictionary mapping candidate string representations to their calculated weights/scores.
        """
        weights: Dict[str, float] = {}
        for key, candidate_obj in candidates.items():
            tokens: List[str] = candidate_obj.lexical_form # Use lemmas for ranking
            # Sum scores of constituent tokens, defaulting to 0 for tokens not in word_scores
            score: float = sum(word_scores.get(token, 0.0) for token in tokens)
            if normalized and tokens: # Avoid division by zero for empty candidates (if any)
                score /= len(tokens)
            weights[key] = score
        return weights


class CandidateFilter:
    """
    Filters candidate keyphrases to remove duplicates or near-duplicates.

    This class uses a FastText model to calculate semantic similarity (Word Mover's Distance)
    between candidates and filters out lower-ranked candidates that are too similar
    to higher-ranked ones.
    """
    def __init__(self, ft_model: Optional[FastText]) -> None:
        """
        Initializes the CandidateFilter.

        :param ft_model: gensim.models.FastText or None, a pre-trained FastText model instance.
                         If None, filtering operations will be skipped.
        """
        self.ft_model: Optional[FastText] = ft_model

    def remove_duplicates(self, candidates: Dict[str, Candidate], weights: Dict[str, float], threshold: Optional[float] = None) -> None:
        """
        Filters candidates by removing duplicates based on Word Mover's Distance (WMD).

        The method iterates through pairs of candidates, sorted by their weights.
        If the WMD between two candidates is below the given threshold, the lower-ranked
        candidate in the pair is marked for removal.
        Modifies the `candidates` and `weights` dictionaries in place.

        :param candidates: dict, a dictionary of Candidate objects, keyed by their string representation.
                           This dictionary will be modified.
        :param weights: dict, a dictionary of candidate weights/scores. This dictionary will be modified.
        :param threshold: float or None, the WMD threshold. If None, no filtering is performed.
                          Candidates with WMD < threshold are considered duplicates.
        """
        if threshold is None:
            logging.info("No threshold provided for duplicate removal. Skipping.")
            return

        if not self.ft_model:
            logging.warning("FastText model not available in CandidateFilter. Skipping duplicate removal.")
            return

        # Sort candidate keys by their weights in descending order
        top_keys: List[str] = sorted(weights, key=lambda k: weights[k], reverse=True)
        keys_to_delete: Set[str] = set() # Stores keys of candidates to be removed
        reasons_to_delete: List[Tuple[str, str, float]] = [] # For logging purposes

        # Compare all unique pairs of candidates
        for (k1, k2) in combinations(top_keys, 2):
            # Skip if either candidate is already marked for deletion
            if k1 in keys_to_delete or k2 in keys_to_delete:
                continue
            
            # k1 is always higher ranked than k2 due to sorted order of top_keys

            try:
                # Ensure candidates exist and have lexical forms for WMD calculation
                if k1 not in candidates or k2 not in candidates:
                    continue
                
                doc1_lexical: List[str] = candidates[k1].lexical_form
                doc2_lexical: List[str] = candidates[k2].lexical_form

                if not doc1_lexical or not doc2_lexical: # Skip if either candidate has no lexical form
                    continue
                
                # Calculate Word Mover's Distance
                # Ensure ft_model.wv is available (FastText model structure)
                if not hasattr(self.ft_model, 'wv'):
                    logging.error("FastText model does not have 'wv' attribute for word vectors. Cannot calculate WMD.")
                    return # Or break, if this is a critical failure for all pairs
                
                distance: float = self.ft_model.wv.wmdistance(doc1_lexical, doc2_lexical)

                # If distance is below threshold, mark the lower-ranked candidate (k2) for deletion
                if distance < threshold:
                    keys_to_delete.add(k2)
                    reasons_to_delete.append((k2, k1, distance))
            except Exception as e:
                # Log errors during WMD calculation, e.g., if words are not in model vocabulary
                logging.error(f"Error calculating WMD between '{k1}' and '{k2}': {e}")
                continue

        # Perform the actual deletion of marked candidates
        if keys_to_delete:
            logging.info(f"Removing {len(keys_to_delete)} duplicate candidates.")
        for k_del_key in keys_to_delete: # Renamed k to k_del_key to avoid conflict with outer scope if any
            if k_del_key in candidates:
                del candidates[k_del_key]
            if k_del_key in weights:
                del weights[k_del_key]

        # Log reasons for deletion for transparency
        for (k_del, k_kept, dist) in reasons_to_delete:
            # Score of k_del might be unavailable if it was already deleted by another pair,
            original_score_del_val: Any = next((w for k_orig, w in zip(top_keys, map(weights.get, top_keys)) if k_orig == k_del), 'N/A')
            logging.info(f"Removing '{k_del}' (original score: {original_score_del_val}) due to similarity (WMD: {dist:.4f}) with '{k_kept}' (score: {weights.get(k_kept, 'N/A')})")


class FastTextRank(BaseRank):
    """
    Implements a keyword extraction algorithm based on TextRank, utilizing FastText
    for semantic similarity checks (Word Mover's Distance) during candidate filtering.
    Orchestrates helper components for text processing, graph handling, and candidate management.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initializes the FastTextRank extractor.

        :param model_path: str or None, path to a pre-trained FastText model file.
                           If None, duplicate removal via WMD will be skipped.
        """
        super().__init__()
        self.model: Optional[FastText] = None
        if model_path:
            try:
                self.model = FastText.load(model_path)
                logging.info(f"FastText model loaded successfully from {model_path}.")
            except Exception as e:
                logging.error(f"Error loading FastText model from {model_path}: {e}")
                raise RuntimeError(f"Failed to load FastText model from {model_path}")
        else:
            logging.warning("No FastText model path provided. Candidate filtering via WMD will be skipped.")
        
        # Initialize helper components
        self.graph_handler: GraphHandler = GraphHandler()
        self.graph: nx.Graph = self.graph_handler.get_graph() # Initialize self.graph from BaseRank
        self.candidate_extractor: CandidateExtractor = CandidateExtractor(self.add_candidate)
        self.candidate_ranker: CandidateRanker = CandidateRanker()
        
        self.candidate_filter: Optional[CandidateFilter]
        if self.model:
            self.candidate_filter = CandidateFilter(self.model)
        else:
            self.candidate_filter = None 
            logging.info("CandidateFilter not initialized as FastText model is unavailable.")


    def process_text(self, text: str) -> None:
        """
        Processes the input text using `TextProcessor`.
        Populates `self.sentences` with a list of Sentence objects.

        :param text: str, the raw input text.
        """
        logging.info("Processing text...")
        doc_obj = TextProcessor.process(text=text, max_length=10**6) 
        self.sentences: List[Sentence] = doc_obj.sentences # Ensure self.sentences is typed
        logging.info(f"Text processed into {len(self.sentences)} sentences.")


    def build_word_graph(self, window: int, pos: Set[str]) -> nx.Graph:
        """
        Builds the word graph using the configured `GraphHandler`.
        Populates `self.graph`. Required by `BaseRank`.

        :param window: int, the co-occurrence window size.
        :param pos: set[str], a set of valid Part-of-Speech tags.
        :return: networkx.Graph, the constructed word graph.
        """
        if not hasattr(self, 'sentences') or not self.sentences:
            logging.error("Sentences not processed. Call process_text() before build_word_graph().")
            return self.graph_handler.get_graph() # Return current (likely empty) graph
        
        logging.info(f"Building word graph with window={window}, pos={pos}...")
        self.graph = self.graph_handler.build_word_graph(self.sentences, window=window, pos=pos)
        logging.info(f"Word graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        return self.graph


    def rank_candidates(self, normalized: bool = False, window: int = 4, pos: Optional[Set[str]] = None, filter_threshold: Optional[float] = None) -> None:
        """
        Orchestrates the candidate ranking process including graph building, PageRank,
        candidate extraction, ranking, and optional filtering.
        Results are stored in `self.weights` and `self.candidates`.

        :param normalized: bool, whether to normalize candidate scores by their length.
        :param window: int, co-occurrence window for graph building.
        :param pos: set[str] or None, valid POS tags. Uses default if None.
        :param filter_threshold: float or None, WMD threshold for duplicate removal.
        """
        if not hasattr(self, 'sentences') or not self.sentences:
            logging.error("Sentences not processed. Call process_text() before rank_candidates().")
            return

        # Use default POS tags if None is provided
        effective_pos: Set[str] = pos if pos is not None else {'NOUN', 'ADJ'}

        logging.info("Starting candidate ranking process...")
        self.build_word_graph(window=window, pos=effective_pos)
        if not self.graph or self.graph.number_of_nodes() == 0:
            logging.warning("Graph building resulted in an empty or invalid graph. Cannot rank candidates.")
            self.weights = {} 
            return

        logging.info("Calculating PageRank for words...")
        word_scores: Dict[str, float] = self.graph_handler.calculate_pagerank() 
        if not word_scores:
            logging.warning("No word scores calculated (PageRank returned empty). Cannot rank candidates.")
            self.weights = {}
            return
        
        keywords: List[str] = sorted(word_scores, key=word_scores.get, reverse=True)
        logging.info(f"PageRank calculated for {len(word_scores)} words.")

        logging.info("Extracting candidate sequences...")
        self.candidate_extractor.extract_sequences(self.sentences, keywords)
        logging.info(f"{len(self.candidates)} initial candidates extracted.")

        logging.info("Ranking extracted candidates...")
        self.weights = self.candidate_ranker.rank(self.candidates, word_scores, normalized)
        logging.info(f"Ranking complete for {len(self.weights)} candidates.")
        
        if filter_threshold is not None:
            if self.candidate_filter:
                logging.info(f"Filtering duplicate candidates with threshold={filter_threshold}...")
                self.candidate_filter.remove_duplicates(self.candidates, self.weights, threshold=filter_threshold)
                logging.info(f"Filtering complete. {len(self.candidates)} candidates remaining.")
            else:
                logging.warning("Filter threshold provided, but CandidateFilter is not available (e.g., FastText model missing). Skipping filtering.")
        else:
            logging.info("Duplicate candidate filtering skipped as no threshold was provided.")
        
        logging.info("Candidate ranking process finished.")
