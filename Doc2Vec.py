import gensim
import numpy
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from typing import Sequence, Iterable, Dict, List
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec

# The aim of this generator function is to read the corpus file and create and yield a TaggedDocument made up of the
# tokenized words of each line in the corpus file, and also be paired with its integer ID tag which will represents
# the corpus number (e.g. either corpus 1 or corpus 2)

def read_corpus(wordsToEmbed: Path, corpusId: int) -> Iterable[TaggedDocument]:
    with open(file=wordsToEmbed, mode='r', encoding= 'utf8') as f:
        for line in f.readlines():
            tokens = gensim.utils.simple_preprocess(line)
            yield TaggedDocument(tokens, [corpusId])


# This generator function will take a list of paths to the corpus files, and return the document which consists of
# a list of tokenized strings of each line as an iterable
def loadDocuments(corpusPaths: Sequence[Path]) -> Iterable[TaggedDocument]:
    for index, path in enumerate(corpusPaths, start=1):
        yield from read_corpus(path, index)


CorpusId = int
Embedding = np.ndarray


class SentenceEmbeddings:
    def __init__(self, targetWord: str, mapping: Dict[CorpusId, Sequence[Embedding]]):
        self.targetWord = targetWord
        self.mapping = mapping

    # This is how we will be able to split the sentence embeddings based on their corpus number
    # (e.g. corpus 1 or corpus 2) after we combine them to train the Doc2Vec model
    def fromCorpus(self, corpusId: CorpusId):
        return self.mapping.get(corpusId)


class ClusterDistance:
    def __init__(self, targetWord: str, clusterDistance: float):
        self.targetWord = targetWord
        self.clusterDistance = clusterDistance


class CorpusDocument:
    def __init__(self, sentence: List[str], corpusId: int):
        self.sentence = sentence
        self.corpusId = corpusId

    def __str__(self):
        return '(%s, %s)' % (self.sentence, self.corpusId)

# this function will find the documents in the passed TaggedDocument which contain the
# target word and return them as an iterable.
def getDocumentsContaining(word: str, corpuses: Sequence[TaggedDocument]) -> Iterable[CorpusDocument]:
    for taggedDocument in corpuses:
        if word in taggedDocument.words:
            yield CorpusDocument(taggedDocument.words, taggedDocument.tags[0])


Word = str

# this generator function takes the path of the target words file and returns the words as an iterable
def readTargetWords(wordsToEmbed: Path) -> Iterable[Word]:
    with open(file=wordsToEmbed, mode='r', encoding='utf8') as f:
        for targetWord in f.readlines():
            yield targetWord.rstrip()

# This function takes the model and the created TaggedDocuments of a language along with the path of the target
# words file. Sentence embeddings are then created for each word where upon the embeddings are split based on
# their corpus number and clustered separately using the KMeans algorithm. The euclidean distance of the
# clustered centre for each corpus will be calculated and returned as an iterable
def getAllSentenceEmbeddings(model: Doc2Vec, taggedDocuments: Sequence[TaggedDocument], wordsToEmbed: Path) -> Iterable[SentenceEmbeddings]:
    sentenceEmbeddings = []
    for word in readTargetWords(wordsToEmbed):
        word = word.lower()
        arr = []

        # creating the sentence embeddings for each target word
        for corpusDocument in getDocumentsContaining(word, taggedDocuments):
            embedding = model.infer_vector(corpusDocument.sentence)
            arr.append((corpusDocument.corpusId, embedding))

        embeddingsPerCorpus = {}
        for corpusId, embedding in arr:
            embeddingsPerCorpus.setdefault(corpusId, []).append(embedding)
        sentenceEmbeddings.append(SentenceEmbeddings(word, embeddingsPerCorpus))
    yield from sentenceEmbeddings


# This function will take the paths of both the corpus and the target words text files, and
# train the Doc2Vec model and create sentence embeddings for each target word.
# Where upon the embeddings from the two corporas will be clustered separately using the KMeans algorithm,
# where the euclidean distance between the two cluster centres will be calculated.
def getClusterCentreDistances(wordsToEmbed: Path, corpora: Sequence[Path]) -> Sequence[ClusterDistance]:
    taggedDocuments = list(loadDocuments(corpora))

    max_epochs = 30  # the total number of iterations over the corpus
    vec_size = 50  # the dimensions of our sentence embeddings
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab(taggedDocuments)

    # the training of our Doc2Vec model
    for epoch in range(max_epochs):
        model.train(taggedDocuments,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate to avoid decay
        model.min_alpha = model.alpha

    # the embeddings will be split based on their corpus number and clustered separately, the euclidean
    # distance of the cluster centre of each corpus will be calculated and returned as an iterable
    for sentenceEmbedding in getAllSentenceEmbeddings(model, taggedDocuments, wordsToEmbed):

        kMeansCorpus1 = KMeans(n_clusters=1, max_iter=300, n_init=10, random_state=0).fit(sentenceEmbedding.fromCorpus(1))
        kMeansCorpus2 = KMeans(n_clusters=1, max_iter=300, n_init=10, random_state=0).fit(sentenceEmbedding.fromCorpus(2))

        a = kMeansCorpus1.cluster_centers_[0]
        b = kMeansCorpus2.cluster_centers_[0]
        clusterDistance = numpy.linalg.norm(a - b)  # euclidean distance calculation
        yield ClusterDistance(sentenceEmbedding.targetWord, clusterDistance)


def main():

    englishWordsToEmbed = Path("trial_data_public/targets/english.txt")
    englishCorpora = [
        Path("trial_data_public/corpora/english/corpus1/corpus.txt"),
        Path("trial_data_public/corpora/english/corpus2/corpus.txt")
    ]
    swedishWordsToEmbed = Path("trial_data_public/targets/swedish.txt")
    swedishCorpora = [
        Path("trial_data_public/corpora/swedish/corpus1/corpus.txt"),
        Path("trial_data_public/corpora/swedish/corpus2/corpus.txt")
    ]

    germanWordsToEmbed = Path("trial_data_public/targets/german.txt")
    germanCorpora = [
        Path("trial_data_public/corpora/german/corpus1/corpus.txt"),
        Path("trial_data_public/corpora/german/corpus2/corpus.txt")
    ]

    latinWordsToEmbed = Path("trial_data_public/targets/latin.txt")
    latinCorpora = [
        Path("trial_data_public/corpora/latin/corpus1/corpus.txt"),
        Path("trial_data_public/corpora/latin/corpus2/corpus.txt")
    ]

    print("ENGLISH WORDS")
    print("********************")
    for i in getClusterCentreDistances(englishWordsToEmbed, englishCorpora):
        print(i.targetWord)
        print(i.clusterDistance)

    print("\n SWEDISH WORDS")
    print("********************")
    for i in getClusterCentreDistances(swedishWordsToEmbed, swedishCorpora):
        print(i.targetWord)
        print(i.clusterDistance)


    print("\n GERMAN WORDS")
    print("********************")
    for i in getClusterCentreDistances(germanWordsToEmbed, germanCorpora):
        print(i.targetWord)
        print(i.clusterDistance)

    print("\n LATIN WORDS")
    print("********************")
    for i in getClusterCentreDistances(latinWordsToEmbed, latinCorpora):
        print(i.targetWord)
        print(i.clusterDistance)

main()
