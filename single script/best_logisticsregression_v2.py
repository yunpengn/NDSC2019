

import pandas as pd
import re

train_df = pd.read_csv('../data/train.csv', sep = ",")
train_df = train_df[['title','Category']]
test_df = pd.read_csv('../data/test.csv', sep = ",")
test_df = test_df[['title', 'itemid']]



def filter_punctuation(line):
    p = re.compile(r'[^a-zA-Z]')
    return re.sub(p,' ', line)

import string
train_df['title'] = train_df.title.map(
    lambda x: filter_punctuation(x)
)
test_df['title'] = test_df.title.map(
    lambda x: filter_punctuation(x)
)


# In[64]:

X_train = train_df['title']
Y_train = train_df['Category']

size = int(0.8 * len(X_train))
X_train = X_train[: size]
Y_train = Y_train[: size]
X_test_id = test_df['itemid']
X_test = test_df['title']

# In[65]:


from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words = 'english')
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(training_data,Y_train)


# In[68]:

predictions = lr.predict(testing_data)

from joblib import dump, load
dump(lr, '../data/lr_model.joblib')

result_dict = {'itemid':X_test_id, 'Category': predictions}
result = pd.DataFrame(data = result_dict)
result.to_csv("./output_lr.csv",index=False)
