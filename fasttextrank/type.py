from typing import NamedTuple, List


class Sentence(NamedTuple):
    """
    a data model for a sentence implemented using NamedTuple
    """
    words: List[str]
    pos: List[str]
    lemmas: List[str]
    length: int


class Document(NamedTuple):
    """
    a data model for document implemented using NamedTuple
    """
    sentences: List[Sentence]
