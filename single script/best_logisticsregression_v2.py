

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

X = train_df['title']
Y = train_df['Category']

size = int(0.8 * len(X))
X_train = X[: size]
Y_train = Y[: size]
X_validate = X[size+1:]
Y_validate = Y[size+1:]
X_test_id = test_df['itemid']
X_test = test_df['title']

# In[65]:


from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words = 'english')
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)
validate_data = count_vector.transform(X_validate)

from joblib import dump, load
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(training_data,Y_train)

# lr = load("../data/lr_model.joblib")
predictions = lr.predict(testing_data)
validate_predictions = lr.predict(validate_data)
print(str(sum(validate_predictions == Y_validate)/ len(validate_predictions)))

dump(lr, '../data/lr_model.joblib')

result_dict = {'itemid':X_test_id, 'Category': predictions}
result = pd.DataFrame(data = result_dict)
result.to_csv("../data/output_lr.csv",index=False)
