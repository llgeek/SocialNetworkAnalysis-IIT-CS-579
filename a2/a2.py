# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("??necronomicon?? geträumte sünden.<br>Hi", True)
    array(['necronomicon', 'geträumte', 'sünden.<br>hi'], 
          dtype='<U13')
    >>> tokenize("??necronomicon?? geträumte sünden.<br>Hi", False)
    array(['necronomicon', 'geträumte', 'sünden', 'br', 'hi'], 
          dtype='<U12')
    """
    ###TODO
    ###pass

    if keep_internal_punct == False:
        return np.array(re.findall('[\w_]+', doc.lower()))
    else:
        return np.array(re.findall('[\w_][^\s]*[\w_]|[\w_]', doc.lower()))



def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    ###pass
    for tk in tokens:
        feats["token="+tk] += 1




def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO
    ###pass
    for i in range(len(tokens)-k+1):
        comb = combinations(tokens[i:i+k], 2)
        for c in comb:
            feats['token_pair='+c[0]+'__'+c[1]] +=1




neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO
    ###pass
    feats['neg_words'] = 0
    feats['pos_words'] = 0
    for t in tokens:
        if t.lower() in neg_words:
            feats['neg_words'] += 1
        if t.lower() in pos_words:
            feats['pos_words'] += 1



def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    ###pass
    feats = defaultdict(lambda: 0)
    for fun in feature_fns:
        fun(tokens, feats)

    return (list(sorted(feats.items())))



def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
    ###pass
    feats_list = list(defaultdict(lambda: 0))
    feats_freq = defaultdict(lambda: 0)

    row = []
    col = []
    data = []

    for tokens in tokens_list:
        feats_list.append(featurize(tokens, feature_fns))
    if vocab == None:
        for feats in feats_list:
            for f, v in feats:
                if v != 0:
                    feats_freq[f] += 1
        id = 0
        vocab = dict()
        for f, v in sorted(feats_freq.items()):
            if v >= min_freq:
                vocab[f] = id
                id += 1


    # for tokens in tokens_list:
    #     feats = defaultdict(lambda: 0)
    #     for fun in feature_fns:
    #         fun(tokens, feats)
    #     #feats_list.append(dict((k,v) for k, v in feats.items() if v >= min_freq))
    #     feats_list.append(feats)
    #     for f in feats:
    #         if feats[f] != 0: 
    #             feats_freq[f] += 1

    # featurelist = sorted(list(k for k in feats_freq if feats_freq[k] >= min_freq))

    # if vocab == None:
    #     vocab = dict((k, featurelist.index(k)) for k in featurelist)
    #     featurelist = set(featurelist)
    # else:
    #     featurelist = set(featurelist) - (set(featurelist) - set(k for k in vocab))

    for i in range(0, len(feats_list)):
    #for feats in feats_list:
        for fkey, v in feats_list[i]:
            #if fkey in featurelist and feats_list[i][fkey] != 0:
            if fkey in vocab:
                row.append(i)
                col.append(vocab[fkey])
                data.append(v)

    return csr_matrix((data, (row, col)), shape=(len(feats_list), len(vocab))), vocab



def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    ###pass
    kf = KFold(len(labels), k)
    accuracies = []

    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        clf.fit(X_train, y_train)
        accuracies.append(accuracy_score(clf.predict(X_test), y_test))

    return np.mean(accuracies)




def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    ###TODO
    ###pass
    results = []

    feature_fns_enum = []
    for i in range(1, len(feature_fns)+1):
        for fea_comb in combinations(feature_fns, i):
            feature_fns_enum.append([*fea_comb])

    for punct in punct_vals:
        tokens_list = [tokenize(doc, punct) for doc in docs]
        for fns in feature_fns_enum:
            for min_freq in min_freqs:
                X, vocab = vectorize(tokens_list, fns, min_freq)
                accuracy = cross_validation_accuracy(LogisticRegression(), X, labels, 5)
                results.append({'features': fns, 'punct': punct, 'accuracy': accuracy, 'min_freq': min_freq})

    return sorted(results, key = lambda x: (-x['accuracy'], -x['min_freq']))



def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    ###pass
    X = range(0, len(results))
    results = sorted(results, key = lambda x : x['accuracy'])
    Y = [y['accuracy'] for y in results]
    #plt.figure(figsize = (3, 3))
    plt.plot(X, Y, 'b-')
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.savefig('accuracies.png')
    #plt.show()
    



def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO
    ###pass
    # setting = ['punct', 'features', 'min_freq']
    settingaccuracies = defaultdict(lambda: 0.)
    settingfreq = defaultdict(lambda: 0)
    for r in results:
        settingaccuracies['punct' + '=' + str(r['punct'])] += r['accuracy']
        settingfreq['punct' + '=' + str(r['punct'])] += 1
        fnname = "features="
        for fn in r['features']:
            fnname += " " + fn.__name__
        settingaccuracies[fnname] += r['accuracy']
        settingfreq[fnname] += 1
        settingaccuracies['min_freq' + '=' + str(r['min_freq'])] += r['accuracy']
        settingfreq['min_freq' + '=' + str(r['min_freq'])] += 1

        # for s in setting:
        #     settingaccuracies[s + '=' + str(r[s])] += r['accuracy']
        #     settingfreq[s + '=' + str(r[s])] += 1

    for key in settingaccuracies:
        settingaccuracies[key] /= settingfreq[key]

    return[(v, k) for k, v in sorted(settingaccuracies.items(), key = lambda x: -x[1])]



def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    ###pass
    tokens_list = []
    clf = LogisticRegression()

    for doc in docs:
        tokens_list.append(tokenize(doc, best_result['punct']))
    X, vocab = vectorize(tokens_list, best_result['features'], best_result['min_freq'])
    clf.fit(X, labels)

    return clf, vocab



def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO
    ###pass
    coef = clf.coef_[0]
    topids = []
    top_coef = []
    if label == 1:
        topids = np.argsort(coef)[::-1][:n]
        top_coef = coef[topids]  #coefficient
    else:
        topids = np.argsort(coef)[:n]
        top_coef = coef[topids] * -1

    top_coef_terms = np.array(sorted(vocab.items(), key=lambda x: x[1]))[topids]  #feature name
    
    return [x for x in zip(top_coef_terms, top_coef)]

    #return [(list(vocab.keys())[list(vocab.values()).index(id)], coef[id]) for id in topids]
    #return sorted([(k, coef[vocab[k]]) for k in vocab if vocab[k] in topids], key = lambda x: x[1])



def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    ###pass
    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    tokens_list = []
    for doc in test_docs:
        tokens_list.append(tokenize(doc, best_result['punct']))
    
    X_test, vocab_ = vectorize(tokens_list, best_result['features'], best_result['min_freq'], vocab)

    return test_docs, test_labels, X_test




def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    ###pass
    predict_labels = clf.predict(X_test)
    different_labels = np.where(predict_labels != test_labels)[0]  #different prediction indice

    predict_prob = clf.predict_proba(X_test)
    wrong_pred = predict_prob[different_labels]

    ids = np.argsort(np.amax(wrong_pred, axis = 1))[::-1][:n]   #first maximum n indice


    for i in range(0, n):
        docid = different_labels[ids[i]]

        print('truth=' + str(test_labels[docid]) + ' predicted=' + str(predict_labels[docid]) + ' proba=' + str(np.max(predict_prob[docid])))
        print(test_docs[docid])
        print()





def main():
    """
    Put it all together.
    ALREADY DONE.
    """

    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)
    


if __name__ == '__main__':
    main()
