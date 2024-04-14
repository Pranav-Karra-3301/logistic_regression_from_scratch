# Logistic Regression from Scratch

https://img.shields.io/badge/-pandas-navy?logo=pandas

https://img.shields.io/badge/-scikit--learn-blue?logo=scikitlearn

# Code Walk-Through

### Importing Libraries

```python
import numpy as np 
from numpy import log,dot,e,shape
import matplotlib.pyplot as plt
```

> We are gonna be using sklearn’s make_classification dataset with 4 features
> 

```python
from sklearn.datasets import make_classification
X,y = make_classification(n_features = 4,n_classes=2)
from sklearn.model_selection import train_test_split  
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.1)

print(X_tr.shape, X_te.shape)
```

<aside>
💡 Running the above code we get:
**(90, 4) (10, 4)**

</aside>

The output basically means The training data (**`X_tr`**) consists of 90 samples, each with 4 features, while the testing data (**`X_te`**) consists of 10 samples, also with 4 features each

---

### Standardization

Standardization is a technique used to scale the features of your data so that they have a **mean of zero and a standard deviation of one**. This process helps to bring all features onto a similar scale, making it easier for machine learning algorithms to process them.

Say $X$ is feature in your dataset, the standardized value $Z$ is calculated as:

$$
Z = \frac{X - \mu}{\sigma}
$$

Here $\mu$ is the mean of the feature $X$ and $\sigma$ is the standard deviation. 

```python
def standardize(X_tr):
    for i in range(shape(X_tr)[1]):
        X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])
```

---

### Initializing the Parameters