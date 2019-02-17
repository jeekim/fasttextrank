import spacy
from fasttextrank.type import Sentence, Document


class TextProcessor(object):
    """
    class for NLP processor
    """
    @staticmethod
    def process(text, max_length=None):
        """
        given a text, call spacy for sentence splitting, lemmatisation, and POS tagging
        :param text:
        :param max_length:
        :return: a document with processed sentences
        """
        nlp = spacy.load('en', max_length=max_length)
        doc = nlp(text)

        sentences = []
        for sent in doc.sents:
            sentences.append({
                "words": [token.text for token in sent],
                "lemmas": [token.lemma_ for token in sent],
                "pos": [token.pos_ for token in sent],
            })

        return Document(sentences=[Sentence(words=s['words'], pos=s['pos'], lemmas=s['lemmas'], length=len(s['words']))
                                   for s in sentences])
