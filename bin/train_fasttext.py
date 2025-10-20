from gensim.models import FastText
import fasttextrank.config as config
import argparse


def main():
    """
    script to training fastText word embedding model
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input-file', required=False, default=config.DATA_FILE,
                        help='input data file for training')
    parser.add_argument('-m', '--model-file', required=False, default=config.MODEL_FILE,
                        help='model output name')
    parser.add_argument('-s', '--embedding-size', required=False, type=int, default=config.EMBEDDING_SIZE,
                        help='A size of word embedding vector')

    args = parser.parse_args()

    model = FastText(size=args.embedding_size, sg=1)
    model.build_vocab(corpus_file=args.input_file)

    total_words = model.corpus_total_words
    model.train(corpus_file=args.input_file, total_words=total_words, epochs=5)

    model.save(args.model_file)


if __name__ == '__main__':
    main()
