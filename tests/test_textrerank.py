import fasttextrank.config as config

from fasttextrank.fasttextrank import FastTextRank

text1 = "We discuss the thesis of selective representing-the idea that the contents of the mental representations" \
       " had by organisms are highly constrained by the biological niches within which the organisms evolved. While" \
       " such a thesis has been defended by several authors elsewhere, our primary concern here is to take up the" \
       " issue of the compatibility of selective representing and realism. We hope to show three things. First, that" \
       " the notion of selective representing is fully consistent with the realist idea of a mind-independent world." \
       " Second, that not only are these two consistent, but that the latter (the realist conception of a" \
       " mind-independent world) provides the most powerful perspective from which to motivate and understand the" \
       " differing perceptual and cognitive profiles themselves."

text2 = "Earth observation missions have recently attracted a growing interest, mainly due to the large number of" \
        " possible applications capable of exploiting remotely sensed data and images. Along with the increasEarth" \
        " observation missions have recently attracted a growing interest, mainly due to the large number of" \
        " possible applications capable of exploiting remotely sensed data and images. Along with the increase of" \
        " market potential, the need arises for the protection of the image products. Such a need is a very" \
        " crucial one, because the Internet and other public/private networks have become preferred means of data" \
        " exchange. A critical issue arising when dealing with digital image distribution is copyright protection." \
        " Such a problem has been largely addressed by resorting to watermarking technology. A question that" \
        " obviously arises is whether the requirements imposed by remote sensing imagery are compatible with" \
        " existing watermarking techniques. On the basis of these motivations, the contribution of this work" \
        " is twofold: assessment of the requirements imposed by remote sensing applications on watermark-based" \
        " copyright protection, and modification of two well-established digital watermarking techniques to meet" \
        " such constraints. More specifically, the concept of near-lossless watermarking is introduced and two" \
        " possible algorithms matching such a requirement are presented. Experimental results are shown to measure" \
        " the impact of watermark introduction on a typical remote sensing application, i.e., unsupervised image" \
        " classification."


def test_fasttextrrank1():
    """
    """
    extractor = FastTextRank(model=config.MODEL_FILE)
    extractor.process_text(text=text1)
    extractor.build_word_graph(window=2, pos=config.VALID_POSTAGS)
    extractor.rank_candidates(normalized=False)
    extractor.remove_duplicate_candidates(threshold=0.1)

    key_phrases = [k for k in extractor.get_nbest(n=5)]

    assert key_phrases == [
        'our primary concern',
        'cognitive profiles themselves',
        'realist idea',
        'realist conception',
        'biological niches']


def test_fasttextrank2():
    """
    """
    extractor = FastTextRank(model=config.MODEL_FILE)
    extractor.process_text(text=text2)
    extractor.build_word_graph(window=2, pos=config.VALID_POSTAGS)
    extractor.rank_candidates(normalized=False)
    extractor.remove_duplicate_candidates(threshold=0.1)

    key_phrases = [k for k in extractor.get_nbest(n=5)]
    print(key_phrases)

    assert key_phrases == [
        'typical remote sensing application',
        'digital image distribution',
        'unsupervised image classification',
        'possible applications capable',
        'remote sensing imagery',
    ]


if __name__ == '__main__':
    test_fasttextrrank1()
    test_fasttextrank2()

