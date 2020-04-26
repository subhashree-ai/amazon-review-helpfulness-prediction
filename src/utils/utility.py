import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from plotting import report, plot_roc

def scaleColumns(df, cols_to_scale):
    min_max_scaler = MinMaxScaler()
    for col in cols_to_scale:
        df[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df[col])),columns=[col])
    return df

# This utility will generate the difference between the first review and the corresponding review days difference.
def getReviewTimeDifferenceFromMin(dataset):
    dataset['reviewTime'] = pd.to_datetime(dataset['reviewTime'])
    dataset['firstReviewTime'] = dataset.groupby(['asin'])['reviewTime'].transform(min)
    dataset['review_first_diff'] = (dataset['reviewTime'] - dataset['firstReviewTime']).astype('timedelta64[D]')
    dataset = dataset.drop(columns = ['firstReviewTime', 'reviewTime'])
    
    return dataset

# This utility will count the sentences in the review text
def countReviewSentence(dataset):
    pun_sen = ['.', '!', '?']
    text_col = dataset['reviewText']
    sentence_counts = []
    for i in text_col:
        sentence_count = []
        for j in pun_sen:
            count_a = i.count(j)
            sentence_count.append(count_a)
        sentence_counts.append(sum(sentence_count))
    dataset['reviewSentencesCount'] = sentence_counts
    return dataset

# This utiity will generate the no of characters present in review text.
# This will be used in determining the readability of each review
punctuation = ['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~','``',"''",'--']
def count_characters(dataset):
    reviewcharacters = []
    text_col = dataset['reviewText']
    for i in text_col:
        a = dict(Counter(i))
        b = {k:v for k, v in a.items() if k not in punctuation}
        c = sum(list(b.values()))
        reviewcharacters.append(c)
    dataset['reviewChars'] = reviewcharacters
    return dataset

# Readability of each review (ARI as index to measure)
def readability(dataset):
    wordperSen = []
    charperWord = []
    reviewRead = []
    len_df = len(dataset)
    dataset['reviewTextWordCount'] = dataset['reviewText'].apply(lambda x: len(x.split()))
    a = list(dataset['reviewTextWordCount'])
    b = list(dataset['reviewSentencesCount'])
    c = list(dataset['reviewChars'])
    for i in range(len_df):
        if b[i] == 0:
            wordperSen.append(0)
        else:
            j = a[i] / b[i]
            wordperSen.append(j)
        if a[i] == 0:
            charperWord.append(0)
        else:
            l = c[i] / a[i]
            charperWord.append(l)
        ari = 4.71 * charperWord[i] + 0.5 * wordperSen[i] - 21.43
        reviewRead.append(ari)
    dataset['reviewReadability'] = reviewRead
    return dataset

# drop extra features in the dataset
def drop_extra_features(dataset):
    dataset = dataset.drop(columns = ['asin', 'reviewerName', 'reviewerID','summary','unixReviewTime', 'reviewTextWordCount','overall',  'reviewSentencesCount', 'reviewChars'], axis = 1)
    
    return dataset

def perform_model_optimization(grids, grid_dict):
    # Fit the grid search objects
    print('Performing model optimizations...')
    best_precision = 0.0
    best_clf = 0
    best_gs = ''
    for idx, gs in enumerate(grids):
        start = time.time()
        print('\nEstimator: %s' % grid_dict[idx])
        # Fit grid search
        gs.fit(X_train, y_train)
        # Best params
        print('Best params: %s' % gs.best_params_)
        # Best training data accuracy
        print('Best training accuracy: %.3f' % gs.best_score_)
        # Predict on test data with best params
        y_pred = gs.predict(X_test)
        # Test data accuracy of model with best params
        print('Test set precision score for best params: %.3f ' % precision_score(y_test, y_pred))
        print('Reports :')
        report(y_test, y_pred)
        plot_roc(y_test, y_pred)
        # Track best (highest test accuracy) model
        if precision_score(y_test, y_pred) > best_precision:
            best_precision = precision_score(y_test, y_pred)
            best_gs = gs
            best_clf = idx
        end= time.time()
        print('Total time to perform model optimization {} : {:4f}'.format(grid_dict[idx], (end - start)))
    print('\nClassifier with best test set precision: %s' % grid_dict[best_clf])
    return (best_clf,best_gs)