from fasttextrank.fasttextrank import FastTextRank
import fasttextrank.config as config
import argparse
import json


def print_explanation(explanation):
    """
    Pretty print the explanation
    :param explanation: explanation dictionary
    """
    print("\n" + "="*80)
    print("EXTRACTION EXPLANATION")
    print("="*80)

    # Print graph statistics
    if explanation['graph_stats']:
        stats = explanation['graph_stats']
        print(f"\n[Graph Statistics]")
        print(f"  - Number of nodes (words): {stats['num_nodes']}")
        print(f"  - Number of edges (connections): {stats['num_edges']}")
        print(f"  - Number of sentences: {stats['num_sentences']}")

        if stats['top_words']:
            print(f"\n  Top 10 words by PageRank score:")
            for word, score in stats['top_words']:
                print(f"    {word}: {score}")

    # Print top keyphrases explanation
    print(f"\n[Top Keyphrases]")
    for i, kp in enumerate(explanation['top_keyphrases'], 1):
        print(f"\n{i}. '{kp['keyphrase']}' (score: {kp['total_score']})")
        print(f"   Occurrences in text: {kp['num_occurrences']}")
        print(f"   Word composition:")
        for word_info in kp['word_breakdown']:
            print(f"     - {word_info['word']}: {word_info['pagerank_score']}")

    # Print removed keyphrases
    if explanation['removed_keyphrases']:
        print(f"\n[Removed Keyphrases (duplicates/similar phrases)]")
        for item in explanation['removed_keyphrases'][:10]:  # Show first 10
            print(f"  - '{item['removed']}' removed (distance: {item['distance']})")
            print(f"    Reason: {item['reason']}")

    print("\n" + "="*80 + "\n")


def main():
    """
    given an input file, extracts keywords.

    if a word embedding model is not give, use the one trained on Hulth dataset as default.
    :return:
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input-file', required=True, help='input text file for keyword extraction')
    parser.add_argument('-m', '--model-file', required=False, default=config.MODEL_FILE,
                        help='model file for word embedding')
    parser.add_argument('-n', '--nbest', required=True, type=int, help='number of best keywords')
    parser.add_argument('-d', '--distance', required=False, type=float, default=0.1,
                        help='threshold for phrase distance to remove')
    parser.add_argument('-w', '--window-size', required=True, type=int,
                        help='a max distance between words for adding their edge into a graph.')
    parser.add_argument('-e', '--explain', action='store_true',
                        help='provide detailed explanation of keyword extraction')
    parser.add_argument('--explain-json', action='store_true',
                        help='output explanation in JSON format')

    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as file:
        text = file.read()

        extractor = FastTextRank(model=args.model_file)
        extractor.process_text(text=text)
        extractor.build_word_graph(window=args.window_size, pos=config.VALID_POSTAGS)
        extractor.rank_candidates(normalized=False)
        extractor.remove_duplicate_candidates(threshold=args.distance)

        key_phrases = [k for k in extractor.get_nbest(n=args.nbest)]

        # Print keyphrases
        print(key_phrases)

        # Print explanation if requested
        if args.explain or args.explain_json:
            explanation = extractor.get_explanation(n=args.nbest)

            if args.explain_json:
                print(json.dumps(explanation, indent=2))
            else:
                print_explanation(explanation)


if __name__ == '__main__':
    main()
