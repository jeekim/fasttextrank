import os

# A heuristic for key phrases
#
# We assumes that key phrases only consist of adjectives and nouns.
VALID_POSTAGS = {'ADJ', 'NOUN'}
MODEL_FILE = os.path.join('model', 'hulth' + "." + 'bin')
