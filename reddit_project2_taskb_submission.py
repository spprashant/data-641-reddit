'''
This notebook expects the data to be in the "CL class project materials" folder in the root folder
Install the pandas, numpy, nltk, empath libraries atleast before running this
'''

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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
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

CROWD_TRAIN_DATA = r"CL class project materials/umd_reddit_suicidewatch_dataset_v2/crowd/train/shared_task_posts_24hours.csv"
CROWD_TRAIN_LABEL = r"CL class project materials/umd_reddit_suicidewatch_dataset_v2/crowd/train/crowd_train.csv"
CROWD_TEST_DATA = r"CL class project materials/umd_reddit_suicidewatch_dataset_v2/crowd/test/shared_task_posts_test_24hours.csv"
CROWD_TEST_LABEL = r"CL class project materials/umd_reddit_suicidewatch_dataset_v2/crowd/test/crowd_test.csv"
EXPERT_DATA = r"CL class project materials/umd_reddit_suicidewatch_dataset_v2/expert/expert_posts_24hours.csv"
EXPERT_LABEL = r"CL class project materials/umd_reddit_suicidewatch_dataset_v2/expert/expert.csv"
N_FOR_LLR = 50

RANDOM_FOREST_MAX_DEPTH = None
MLP_HIDDEN_LAYERS = [7, 2]
MLP_LEARNING_RATE = 1e-5
MLP_SOLVER = 'adam'
LOG_REGRESSION_SOLVER = 'liblinear'


'''
Helper functions
'''


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
                                          vocabulary=vocabulary)
    X_features = training_vectorizer.fit_transform(X)
    return X_features, training_vectorizer


'''
Convert text to emotion score features
'''


def line_to_emotion_features(lines, lexicon, categories=["negative_emotion", "swearing_terms", "positive_emotion",
                                                         "pain", "nervousness", "death", "shame", "torment"], normalize=True):
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
Function which gets to top differentiating tokens between two labels
'''


def get_vocab_by_llr(train_data, N=N_FOR_LLR):
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

    vocab = []

    for k, v in ranked[:N]:
        vocab.append(k)

    for k, v in ranked[-N:]:
        vocab.append(k)

    return vocab


def process_data(train_data, training_vectorizer=None, vocab_size=N_FOR_LLR):

    X = train_data['features']
    y = train_data['binary_label']

    if not training_vectorizer:
        if vocab_size:
            vocab = get_vocab_by_llr(train_data, vocab_size)
            X_vec, training_vectorizer = convert_text_into_features(
                X, stopwords_arg=stop_words, analyzefn=whitespace_tokenizer, range=(1, 2), vocabulary=vocab)
        else:
            X_vec, training_vectorizer = convert_text_into_features(
                X, stopwords_arg=stop_words, analyzefn=whitespace_tokenizer, range=(1, 2))
    else:
        X_vec = training_vectorizer.transform(X)

    X_emo = line_to_emotion_features(train_data['post'], lexicon=lexicon, categories=["negative_emotion", "swearing_terms", "positive_emotion",
                                                                                      "pain", "nervousness", "death", "shame", "torment"], normalize=False)
    X_ft = sparse.hstack((X_vec, X_emo))
    #X_ft = X_emo
    return X_ft, y, training_vectorizer


def calculate_stats(X, y, classifiers, kfold):

    def scorer(estimator, x, y):
        y_pred = estimator.predict(x)
        return {'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, pos_label=1),
                'recall': recall_score(y, y_pred, pos_label=1),
                'f1_score': f1_score(y, y_pred, pos_label=1)}

    for clf in classifiers:
        print("Classifier : {}".format(clf.name))
        scores = cross_val_score(
            clf.clf,  X, y, scoring='accuracy', cv=kfold, n_jobs=-1)
        print("accuracy scores = {}, mean = {}, stdev = {}".format(
            scores, np.mean(scores), np.std(scores)), flush=True)
        scores = cross_val_score(
            clf.clf,  X, y, scoring='f1', cv=kfold, n_jobs=-1)
        print("f1 scores = {}, mean = {}, stdev = {}".format(
            scores, np.mean(scores), np.std(scores)), flush=True)


'''
Load crowd train data
'''


def load_crowd_data():
    labels_df = pd.read_csv(
        CROWD_TRAIN_LABEL)
    labels_df.head()

    posts_df = pd.read_csv(
        CROWD_TRAIN_DATA)
    posts_df.head()

    train_data = pd.merge(posts_df, labels_df, on=["user_id"])
    train_data = train_data.dropna()

    train_data["binary_label"] = train_data.label.map(
        {"a": 0, "b": 0, "c": 0, "d": 1})

    train_data['post'] = train_data[[
        'post_title', 'post_body']].agg(' '.join, axis=1)
    tqdm.pandas()
    print('Creating unigrams and bigrams')
    train_data['features'] = train_data['post'].progress_apply(
        convert_lines_to_feature_strings, args=(stop_words, True))

    return train_data


'''
Expert data set load
'''


def load_expert_data():
    labels_df = pd.read_csv(
        EXPERT_LABEL)
    labels_df.head()

    posts_df = pd.read_csv(
        EXPERT_DATA)
    posts_df.head()

    train_data = pd.merge(posts_df, labels_df, on=["user_id"])
    train_data = train_data.dropna()

    train_data["binary_label"] = train_data.label.map(
        {"a": 0, "b": 0, "c": 0, "d": 1})
    train_data['post'] = train_data[[
        'post_title', 'post_body']].agg(' '.join, axis=1)

    tqdm.pandas()

    train_data['features'] = train_data['post'].progress_apply(
        convert_lines_to_feature_strings, args=(stop_words, True))

    return train_data, train_data['user_id']


'''
Load crowd test data
'''


def load_crowd_test_data():
    labels_df = pd.read_csv(
        CROWD_TEST_LABEL)
    labels_df.head()

    posts_df = pd.read_csv(
        CROWD_TEST_DATA)
    posts_df.head()

    train_data = pd.merge(posts_df, labels_df, on=["user_id"])
    train_data = train_data.dropna()

    train_data["binary_label"] = train_data.raw_label.map(
        {"a": 0, "b": 0, "c": 0, "d": 1})
    train_data['post'] = train_data[[
        'post_title', 'post_body']].agg(' '.join, axis=1)
    tqdm.pandas()

    train_data['features'] = train_data['post'].progress_apply(
        convert_lines_to_feature_strings, args=(stop_words, True))

    return train_data, train_data['user_id']


if __name__ == '__main__':
    print('---Loading crowd test and train data---', flush=True)
    crowd_data = load_crowd_data()
    crowd_test_data, cwd_uid = load_crowd_test_data()
    '''
    Training Block
    '''

    X_crowd, y_crowd, training_vectorizer = process_data(
        crowd_data, vocab_size=N_FOR_LLR)

    print("Before resample counts: {}".format(Counter(y_crowd)), flush=True)
    balance = False
    if balance:
        X_resampled, y_resampled = SMOTE().fit_resample(X_crowd, y_crowd)
    else:
        X_resampled, y_resampled = X_crowd, y_crowd
    print("After resample counts: {}".format(Counter(y_resampled)), flush=True)

    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    lr_classifier = LogisticRegression(solver=LOG_REGRESSION_SOLVER)
    nb_classifier = MultinomialNB()
    rf_classifier = RandomForestClassifier(random_state=42)
    mlp = MLPClassifier(solver='adam', alpha=MLP_LEARNING_RATE,
                        hidden_layer_sizes=MLP_HIDDEN_LAYERS, random_state=42)

    CLF = namedtuple('CLF', ['name', 'clf'])
    classifiers = [CLF('Logistic Regression', lr_classifier),
                   CLF('Naive Bayes', nb_classifier),
                   CLF('MultiLayer Perceptron', mlp),
                   CLF('Random Forest Classifier', rf_classifier)]

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    for clf in classifiers:
        clf.clf.fit(X_train, y_train)

    for clf in classifiers:
        print("Classifying test data using {}".format(clf.name), flush=True)
        predicted_labels = clf.clf.predict(X_test)
        print('Accuracy  = {}'.format(
            metrics.accuracy_score(predicted_labels,  y_test)), flush=True)
        for label in [0, 1]:
            print('Precision for label {} = {}'.format(
                label, metrics.precision_score(predicted_labels, y_test, pos_label=label)), flush=True)
            print('Recall    for label {} = {}'.format(
                label, metrics.recall_score(predicted_labels,    y_test, pos_label=label)), flush=True)

    '''Cross Validation block (Training only)'''
    print('---Printing train only cross validation metrics---', flush=True)
    calculate_stats(X_resampled, y_resampled, classifiers, kfold)

    '''
    Test Block
    '''
    print('---Testing against crowd test data---', flush=True)
    X_cwd_test, y_cwd_test, _ = process_data(
        crowd_test_data, training_vectorizer)

    crowd_test_data.head()

    for clf in classifiers:
        print("Classifying test data using {}".format(clf.name))
        predicted_labels = clf.clf.predict(X_cwd_test)
        print('Accuracy  = {}'.format(
            metrics.accuracy_score(predicted_labels,  y_cwd_test)))
        print('F1 score    for label {} = {}'.format(
            label, metrics.f1_score(predicted_labels,    y_cwd_test, pos_label=1)))
        for label in [0, 1]:
            print('Precision for label {} = {}'.format(
                label, metrics.precision_score(predicted_labels, y_cwd_test, pos_label=label)))
            print('Recall    for label {} = {}'.format(label, metrics.recall_score(
                predicted_labels,    y_cwd_test, pos_label=label)))

    print('---User ID level validation of test data---', flush=True)
    '''User level validation'''
    for clf in classifiers:
        print("User level validation for model {}".format(clf.name), flush=True)
        predicted_labels = clf.clf.predict(X_cwd_test)

        user_prediction = {}
        for idx, uid in enumerate(cwd_uid):
            if not user_prediction.get(uid):
                user_prediction[uid] = {'total': 0, 'positive': 0}
            if predicted_labels[idx] == 1:
                user_prediction[uid]['positive'] += 1
            user_prediction[uid]['total'] += 1

        labels_df = pd.read_csv(
            CROWD_TEST_LABEL)
        labels_df.head()

        labels_df['label'] = labels_df.raw_label.map(
            {"a": 0, "b": 0, "c": 0, "d": 1})

        fn, fp, tn, tp = 0, 0, 0, 0
        for key, val in user_prediction.items():
            l = labels_df['label'][labels_df['user_id'] == key].values[0]
            #print (val['positive'] / val['total'])
            if val['positive'] / val['total'] > 0.5:
                if l == 0:
                    fp += 1
                elif l == 1:
                    tp += 1
            elif val['positive'] / val['total'] <= 0.5:
                if l == 0:
                    tn += 1
                elif l == 1:
                    fn += 1

        precision = tp/(fp+tp)
        recall = tp/(fn+tp)

        f1_score = 2 * ((precision * recall) / (precision + recall))

        print("Precision {} Recall {} F1 Score {}".format(
            precision, recall, f1_score), flush=True)
        print('Loading expert data', flush=True)
        expert_data, expert_uid = load_expert_data()

        X_expert, y_expert, _ = process_data(expert_data, training_vectorizer)
        print('Running against expert data')
        for clf in classifiers:
            print("Classifying test data using {}".format(clf.name), flush=True)
            predicted_labels = clf.clf.predict(X_expert)
            print('Accuracy  = {}'.format(
                metrics.accuracy_score(predicted_labels,  y_expert)), flush=True)
            print('F1 score    for label {} = {}'.format(
                label, metrics.f1_score(predicted_labels,    y_expert)), flush=True)
            for label in [0, 1]:
                print('Precision for label {} = {}'.format(
                    label, metrics.precision_score(predicted_labels, y_expert, pos_label=label)), flush=True)
                print('Recall    for label {} = {}'.format(
                    label, metrics.recall_score(predicted_labels,    y_expert, pos_label=label)), flush=True)

    print('---User ID level validation of expert data---', flush=True)
    '''User level validation'''
    for clf in classifiers:
        print("User level validation for model {}".format(clf.name), flush=True)
        predicted_labels = clf.clf.predict(X_expert)

        user_prediction = {}
        for idx, uid in enumerate(expert_uid):
            if not user_prediction.get(uid):
                user_prediction[uid] = {'total': 0, 'positive': 0}
            if predicted_labels[idx] == 1:
                user_prediction[uid]['positive'] += 1
            user_prediction[uid]['total'] += 1

        labels_df = pd.read_csv(
            EXPERT_LABEL)
        labels_df.head()

        labels_df['label'] = labels_df.label.map(
            {"a": 0, "b": 0, "c": 0, "d": 1})

        fn, fp, tn, tp = 0, 0, 0, 0
        for key, val in user_prediction.items():
            l = labels_df['label'][labels_df['user_id'] == key].values[0]
            if val['positive'] / val['total'] > 0.5:
                if l == 0:
                    fp += 1
                elif l == 1:
                    tp += 1
            elif val['positive'] / val['total'] <= 0.5:
                if l == 0:
                    tn += 1
                elif l == 1:
                    fn += 1

        precision = tp/(fp+tp)
        recall = tp/(fn+tp)

        f1_score = 2 * ((precision * recall) / (precision + recall))

        print("Precision {} Recall {} F1 Score {}".format(
            precision, recall, f1_score), flush=True)
