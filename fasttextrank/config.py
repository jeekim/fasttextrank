import os
"""
default values for running scripts (e.g., extracting key words, training word embedding models, etc.)
"""

# A heuristic for key phrases
#
# We assumes that key phrases only consist of adjectives and nouns.
VALID_POSTAGS = {'ADJ', 'NOUN'}

DATA_FILE = os.path.join('data', 'hulth')
MODEL_FILE = os.path.join('model', 'hulth' + "." + 'model')

EMBEDDING_SIZE = 300


