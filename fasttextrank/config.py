import os
from typing import Set, Optional

"""
This module defines default configuration values for the FastTextRank system.

These configurations are used across various scripts for tasks such as 
extracting keywords, training word embedding models, and specifying file paths.
Values can be overridden by environment variables where specified.
"""

# A heuristic for key phrases:
# Assumes that key phrases primarily consist of adjectives and nouns.
VALID_POSTAGS: Set[str] = {'ADJ', 'NOUN'}

# Default path for the data file used in training.
# Can be overridden by the FASTTEXTRANK_DATA_FILE environment variable.
# os.getenv returns Optional[str], but with a default str, it's effectively str.
DATA_FILE: str = os.getenv('FASTTEXTRANK_DATA_FILE', os.path.join('data', 'hulth'))

# Default path for the pre-trained FastText model file.
# Can be overridden by the FASTTEXTRANK_MODEL_FILE environment variable.
MODEL_FILE: str = os.getenv('FASTTEXTRANK_MODEL_FILE', os.path.join('model', 'hulth' + "." + 'model'))

# Default embedding size for the FastText model.
EMBEDDING_SIZE: int = 300
