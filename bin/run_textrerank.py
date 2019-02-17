from fasttextrank.fasttextrank import FastTextRank
import fasttextrank.config as config
import argparse


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

    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as file:
        text = file.read()

        extractor = FastTextRank(model=args.model_file)
        extractor.process_text(text=text)
        extractor.build_word_graph(window=args.window_size, pos=config.VALID_POSTAGS)
        extractor.rank_candidates(normalized=False)
        extractor.remove_duplicate_candidates(threshold=args.distance)

        key_phrases = [k for k in extractor.get_nbest(n=args.nbest)]
        print(key_phrases)


if __name__ == '__main__':
    main()
