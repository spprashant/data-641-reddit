from collections import namedtuple
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from scipy import sparse
from empath import Empath
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
from collections import Counter
import re
import string
import nltk
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE, RandomOverSampler
nltk.download('stopwords')


nlp = spacy.load('en_core_web_sm')
lexicon = Empath()
stop_words = set(stopwords.words('english'))


def normalize_tokens(tokenlist):
    normalized_tokens = [token.lower().replace('_', '+') for token in tokenlist
                         if re.search('[^\s]', token) is not None
                         and not token.startswith("@")
                         ]
    return normalized_tokens


def ngrams(tokens, n):
    return [tokens[i:i+n] for i in range(len(tokens)-n+1)]


def filter_punctuation_bigrams(ngrams):
    punct = string.punctuation
    return [ngram for ngram in ngrams if ngram[0] not in punct and ngram[1] not in punct]


def filter_stopword_bigrams(ngrams, stopwords):
    result = [ngram for ngram in ngrams if ngram[0]
              not in stopwords and ngram[1] not in stopwords]
    return result


def whitespace_tokenizer(line):
    return line.split()


'''
Helper functions for convert text lines to features
'''


def convert_lines_to_feature_strings(line, stopwords, remove_stopword_bigrams=True):
    line = line.translate(str.maketrans('', '', string.punctuation))
    spacy_analysis = nlp(line)
    spacy_tokens = [token.orth_ for token in spacy_analysis]
    normalized_tokens = normalize_tokens(spacy_tokens)

    unigrams = [token for token in normalized_tokens
                if token not in stopwords and token not in string.punctuation]

    bigrams = []
    bigram_tokens = ["_".join(bigram) for bigram in bigrams]
    bigrams = ngrams(normalized_tokens, 2)
    bigrams = filter_punctuation_bigrams(bigrams)
    if remove_stopword_bigrams:
        bigrams = filter_stopword_bigrams(bigrams, stopwords)
    bigram_tokens = ["_".join(bigram) for bigram in bigrams]
    feature_string = " ".join([" ".join(unigrams), " ".join(bigram_tokens)])
    return feature_string


def convert_text_into_features(X, stopwords_arg, analyzefn="word", range=(1, 2), vocabulary=None):
    training_vectorizer = CountVectorizer(stop_words=stopwords_arg,
                                          analyzer=analyzefn,
                                          lowercase=True,
                                          ngram_range=range,
                                          vocabulary=vocabulary)
    X_features = training_vectorizer.fit_transform(X)
    return X_features, training_vectorizer


'''
EMPATH - Convert text to emotion score features
'''


def line_to_emotion_features(lines, lexicon, categories=["negative_emotion", "positive_emotion", "pain", "poor", "disappointment"], normalize=False):
    scores = []
    for line in lines:
        score_dict = lexicon.analyze(
            line, categories=categories, normalize=normalize)
        if score_dict:
            score = [score_dict[cat] for cat in categories]
        else:
            score = [0 for cat in categories]
        scores.append(score)
    # print(scores)
    return sparse.csr_matrix(np.array(scores))


'''
LLR Implementation
'''


def get_vocab_by_llr(train_data):
    pos_counter = Counter()
    for post in train_data['features'][train_data['binary_label'] == 1]:
        for x in post.split(' '):
            pos_counter[x] += 1

    neg_counter = Counter()
    for post in train_data['features'][train_data['binary_label'] == 0]:
        for x in post.split(' '):
            neg_counter[x] += 1

    import llr

    diff = llr.llr_compare(pos_counter, neg_counter)
    ranked = sorted(diff.items(), key=lambda x: x[1])

    N = 5000

    vocab = []

    for k, v in ranked[:N]:
        vocab.append(k)

    for k, v in ranked[-N:]:
        vocab.append(k)
    return vocab


def load_crowd_data():
    labels_df = pd.read_csv(
        "CL class project materials/umd_reddit_suicidewatch_dataset_v2/crowd/train/crowd_train.csv")
    labels_df.head()

    posts_df = pd.read_csv(
        "CL class project materials/umd_reddit_suicidewatch_dataset_v2/crowd/train/shared_task_posts.csv")
    posts_df.head()

    train_data = pd.merge(posts_df, labels_df, on=["user_id"])
    train_data = train_data.drop(["post_id", "timestamp", "subreddit"], axis=1)
    train_data = train_data.dropna()

    train_data["binary_label"] = train_data.label.map(
        {"a": 0, "b": 0, "c": 0, "d": 1})

    train_data['features'] = train_data['post_body'].apply(
        convert_lines_to_feature_strings, args=(stop_words, True))

    vocab = get_vocab_by_llr(train_data)

    X = train_data['features']
    y = train_data['binary_label']

    X_vec, training_vectorizer = convert_text_into_features(
        X, stopwords_arg=stop_words, analyzefn=whitespace_tokenizer, range=(1, 2), vocabulary=vocab)
    X_emo = line_to_emotion_features(X, lexicon=lexicon, categories=[
                                     "negative_emotion", "positive_emotion"], normalize=False)
    X_ft = sparse.hstack((X_vec, X_emo))

    return X_ft, y


def load_expert_data():
    labels_df = pd.read_csv(
        "CL class project materials/umd_reddit_suicidewatch_dataset_v2/expert/crowd_train.csv")
    labels_df.head()

    posts_df = pd.read_csv(
        "CL class project materials/umd_reddit_suicidewatch_dataset_v2/expert/shared_task_posts.csv")
    posts_df.head()


def calculate_stats(X, y, classifiers):

    def scorer(estimator, x, y):
        y_pred = estimator.predict(x)
        return {'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, pos_label=1),
                'recall': recall_score(y, y_pred, pos_label=1),
                'f1_score': f1_score(y, y_pred)}

    for clf in classifiers:
        print("Classifier : {}".format(clf.name))
        scores = cross_validate(
            clf.clf,  X, y, scoring=scorer, cv=kfold, n_jobs=-1)
        print("accuracy scores = {}, mean = {}, stdev = {}".format(
            scores['test_accuracy'], np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])), flush=True)
        print("precision scores = {}, mean = {}, stdev = {}".format(
            scores['test_precision'], np.mean(scores['test_precision']), np.std(scores['test_precision'])), flush=True)
        print("recall scores = {}, mean = {}, stdev = {}".format(
            scores['test_recall'], np.mean(scores['test_recall']), np.std(scores['test_recall'])), flush=True)
        print("f1 scores = {}, mean = {}, stdev = {}".format(
            scores['test_f1_score'], np.mean(scores['test_f1_score']), np.std(scores['test_f1_score'])), flush=True)


if __name__ == '__main__':
    X, y = load_crowd_data()
    print("Before resample counts: {}".format(Counter(y)))
    balance = True
    if balance:
        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y
    print("After resample counts: {}".format(Counter(y_resampled)))

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    lr_classifier = LogisticRegression(solver='liblinear')
    nb_classifier = MultinomialNB()
    rf_classifier = RandomForestClassifier(max_depth=4, random_state=0)
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(7, 2), random_state=1)

    CLF = namedtuple('CLF', ['name', 'clf'])
    classifiers = [CLF('Logistic Regression', lr_classifier),
                   CLF('Naive Bayes', nb_classifier),
                   CLF('MultiLayer Perceptron', mlp),
                   CLF('Random Forest Classifier', rf_classifier)]

    calculate_stats(X_resampled, y_resampled, classifiers)
