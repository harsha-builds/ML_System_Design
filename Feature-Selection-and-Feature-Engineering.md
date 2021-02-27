# Feature Selection and Feature Engineering

## 1. One hot encoding
One hot encoding is a very common technique in feature engineering. It converts categorical variables into a one-hot numeric array.
One hot encoding is very popular when you have to deal with categorical features that have medium cardinality.

### Common problems
- Expansive computation and high memory consumption are major problems with one hot encoding. High numbers of values will create high-dimensional feature vectors. For example, if there are one million unique values in a column, it will produce feature vectors that have a dimensionality of one million.
- One hot encoding is not suitable for Natural Language Processing tasks. Microsoft Word’s dictionary is usually large, and we can’t use one hot encoding to represent each word as the vector is too big to store in memory.

### Best practices
- Depending on the application, some levels/categories that are not important, can be grouped together in the “Other” class.
- Make sure that the pipeline can handle unseen data in the test set.

In Python, there are many ways to do one hot encoding, for example, `pandas.get_dummies` and `sklearn OneHotEncoder`. `pandas.get_dummies` does not “remember” the encoding during training, and if testing data has new values, it can lead to inconsistent mapping. `OneHotEncoder` is a Scikitlearn Transformer; therefore, you can use it consistently during training and predicting.
