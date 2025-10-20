# Refactoring Plan for fastTextRank

This document outlines the plan for refactoring the `fastTextRank` codebase to improve its maintainability, readability, and extensibility.

## Plan Steps

1.  **Enhance Configuration Management.**
    *   Modify `fasttextrank/config.py` and relevant files to allow `DATA_FILE` and `MODEL_FILE` paths to be configurable (e.g., via environment variables or command-line arguments). Default values will be maintained if environment variables are not set.

2.  **Improve Logging Practices.**
    *   Review and adjust logging levels in `fasttextrank/fasttextrank.py` and `fasttextrank/baserank.py` for better clarity (e.g., use `INFO` or `DEBUG` instead of `WARNING` for non-critical messages like candidate removal notifications).

3.  **Strengthen Error Handling.**
    *   Add error handling for critical operations, such as FastText model loading in `fasttextrank/fasttextrank.py`. This includes try-except blocks and raising appropriate exceptions if model loading fails.

4.  **Increase Modularity of `FastTextRank` Class.**
    *   Identify and extract parts of the `FastTextRank` class in `fasttextrank/fasttextrank.py` into smaller, more focused classes.
        *   **`GraphHandler`**: Manages graph creation (`build_word_graph`) and PageRank calculation.
        *   **`CandidateExtractor`**: Handles the selection of longest keyword sequences (`select_longest_keyword_sequences`).
        *   **`CandidateRanker`**: Responsible for ranking the extracted candidate phrases.
        *   **`CandidateFilter`**: Manages the removal of duplicate candidates using Word Mover's Distance (`remove_duplicate_candidates`).
    *   The `FastTextRank` class will be refactored to orchestrate these components.

5.  **Update and Review Unit Tests.**
    *   Review and update tests in `tests/test_textrerank.py` to ensure they cover the refactored code and verify the expected behavior. This includes mocking dependencies like the FastText model and ensuring tests for new components.

6.  **Improve Docstrings and Comments.**
    *   Review and enhance docstrings and comments throughout the `fasttextrank` module for better code understanding and maintainability, ensuring they are up-to-date with the refactored code.

7.  **Introduce Comprehensive Type Hinting.**
    *   Add or improve type hints across the `fasttextrank` module (Python `typing` module) to leverage static analysis tools and improve code readability.

8.  **Create `refactoring_plan.md` (This Document).**
    *   Document the approved refactoring plan in this markdown file in the root of the repository.

9.  **Submit the changes.**
    *   Commit all refactoring changes and the `refactoring_plan.md` file.
