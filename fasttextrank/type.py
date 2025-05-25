from typing import NamedTuple, List


class Sentence(NamedTuple):
    """
    Represents a single sentence with its linguistic annotations.

    This data structure uses NamedTuple for immutability and clear field names.
    It stores the original words, their Part-of-Speech (POS) tags, lemmas,
    and the total number of words in the sentence.
    """
    words: List[str]    # List of original word tokens in the sentence.
    pos: List[str]      # List of Part-of-Speech tags corresponding to each word.
    lemmas: List[str]   # List of lemmatized forms of each word.
    length: int         # The total number of words (tokens) in the sentence.


class Document(NamedTuple):
    """
    Represents a document as a collection of sentences.

    This data structure uses NamedTuple and primarily holds a list of Sentence objects
    that constitute the document.
    """
    sentences: List[Sentence] # A list of Sentence objects representing the document's content.
