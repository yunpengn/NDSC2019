
import pandas as pd

train_df = pd.read_csv('../data/train.csv', sep = ",")
train_df = train_df[['title','Category']]
test_df = pd.read_csv('../data/test.csv', sep = ",")
test_df = test_df[['title', 'itemid']]



import string
train_df['title'] = train_df.title.map(
    lambda x: x.lower().translate(str.maketrans('','', string.punctuation))
)
test_df['title'] = test_df.title.map(
    lambda x: x.lower().translate(str.maketrans('','', string.punctuation))
)


X_train = train_df['title']
Y_train = train_df['Category']
X_test_id = test_df['itemid']
X_test = test_df['title']



from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(stop_words = 'english')
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)



from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, Y_train)



predictions = naive_bayes.predict(testing_data)
result_dict = {'itemid':X_test_id, 'Category': predictions}
result = pd.DataFrame(data = result_dict)
result.to_csv("./output.csv",index=False)

