import spacy
from spacy.language import Language as SpacyLanguage
from spacy.tokens import Doc as SpacyDoc
from typing import List, Dict, Optional

from .type import Sentence, Document


class TextProcessor(object):
    """
    Provides text processing capabilities using spaCy.

    This class is responsible for taking raw text and converting it into a structured
    format (Document object containing Sentence objects) with linguistic annotations
    such as lemmas and Part-of-Speech (POS) tags.
    """
    @staticmethod
    def process(text: str, max_length: Optional[int] = 10**6) -> Document:
        """
        Processes raw text to extract sentences with linguistic annotations.

        It uses spaCy for sentence segmentation, tokenization, lemmatization,
        and Part-of-Speech tagging.

        :param text: str, the input text to process.
        :param max_length: int or None, the maximum text length that spaCy should process.
                           If None, spaCy's default limit is used. Defaults to 1,000,000.
        :return: Document, a Document object containing a list of processed Sentence objects.
        """
        # Load the small English model from spaCy.
        # For other languages or larger models, this would need to be changed.
        nlp: SpacyLanguage = spacy.load('en_core_web_sm') 
        
        # Set spaCy's maximum text length if specified
        if max_length:
            nlp.max_length = max_length
        
        # Process the text with spaCy
        spacy_doc: SpacyDoc = nlp(text)

        # Extract sentence data into a temporary list of dictionaries
        processed_sentences_data: List[Dict[str, List[str]]] = []
        for sent in spacy_doc.sents:
            processed_sentences_data.append({
                "words": [token.text for token in sent],      # Original words
                "lemmas": [token.lemma_ for token in sent],   # Lemmatized forms
                "pos": [token.pos_ for token in sent],        # Part-of-Speech tags
            })

        # Convert the list of dictionaries into Sentence objects and wrap in a Document object
        document_sentences: List[Sentence] = [
            Sentence(
                words=s_data['words'], 
                pos=s_data['pos'], 
                lemmas=s_data['lemmas'], 
                length=len(s_data['words'])
            ) for s_data in processed_sentences_data
        ]
        
        return Document(sentences=document_sentences)
