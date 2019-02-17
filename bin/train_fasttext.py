from gensim.models import FastText
import fasttextrank.config as config
import argparse


def main():
    """
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input-file', required=False, default=config.DATA_FILE,
                        help='')
    parser.add_argument('-m', '--model-file', required=False, default=config.MODEL_FILE,
                        help='')

    args = parser.parse_args()

    model = FastText(size=300, sg=1)
    model.build_vocab(corpus_file=args.input_file)

    total_words = model.corpus_total_words
    model.train(corpus_file=args.input_file, total_words=total_words, epochs=5)

    model.save(args.model_file)


if __name__ == '__main__':
    main()
