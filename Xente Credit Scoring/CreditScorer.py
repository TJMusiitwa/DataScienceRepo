# %% [markdown]
# XENTE Credit Scoring Problem

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Import the data to work with

train = pd.read_csv(
    'C:/Users/Gloria/Desktop/Data Science/Xente Credit Scoring/Train.csv')
test = pd.read_csv(
    'C:/Users/Gloria/Desktop/Data Science/Xente Credit Scoring/Test.csv')
unlinked = pd.read_csv(
    'C:/Users/Gloria/Desktop/Data Science/Xente Credit Scoring/unlinked_masked_final.csv')

# %%
train.head()


# %%
test.head()


# %%
unlinked.head()

# %%
train.isnull().sum()

# %%
unlinked.info()

# %%
train = pd.merge(pd, unlinked, on='TransactionId')

# %%
train.type()

# %%
sns.heatmap(train, annot=True)

# %%
train.drop(['TransactionStartTime', 'BatchId', 'CustomerId', 'Currency', 'CountryCode', 'ProviderId', 'ChannelId', 'IssuedDateLoan', 'PaidOnDate',
            'InvestorId', 'DueDate', 'LoanApplicationId', 'PayBackId', 'ThirdPartyId', 'ProductCategory', 'ProductId'], axis=1, inplace=True)


# %%


# %%
scaler = StandardScaler()

# %%
scaler.fit(train)

# %%
scaled_train = scaler.transform(train)

# %%
train_pca = PCA(n_components=10)

# %%
train_pca.fit(scaled_train)

# %%
fit_pca = train_pca.transform(scaled_train)

# %%
plt.figure(figsize=(10, 8))
plt.scatter(fit_pca[:, 0:9], c=train['IsDefaulted'])

# %%


# %%
train.drop(['CurrencyCode', 'SubscriptionId'], axis=1, inplace=True)

# %%
