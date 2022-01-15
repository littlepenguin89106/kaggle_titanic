# %%
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing as preprocessing


# %%
df_train = pd.read_csv("titanic/train.csv")
df_test = pd.read_csv("titanic/test.csv")
df_data = df_train.append(df_test)
df_data.reset_index(inplace=True,drop=True)

# %%
df_data.info()

# %%
sns.set(context="paper", font="monospace")
sns.set(style="white")
f, ax = plt.subplots(figsize=(10,6))
data_corr = df_data.drop('PassengerId',axis=1).corr()
sns.heatmap(data_corr, ax=ax, vmax=.9, square=True,cmap='YlGnBu')
ax.set_xticklabels(data_corr.index, size=15)
ax.set_yticklabels(data_corr.columns[::-1], size=15)
ax.set_title('train feature corr', fontsize=20)

# %%
sns.countplot(df_data['Sex'], hue=df_data['Survived'])

# %%
sns.countplot(df_data['Embarked'], hue=df_data['Survived'])

# %%
sns.boxplot(y=df_data['Fare'])

# %%
scaler = preprocessing.StandardScaler()
scaler = scaler.fit(df_data['Fare'].values.reshape(-1, 1))
df_data['Fare'] = scaler.transform(df_data['Fare'].values.reshape(-1, 1))

# %%
df_data['Fare'] = df_data['Fare'].fillna(df_data['Fare'].median())

# %%
df_data['Embarked'] = df_data['Embarked'].fillna('S')

# %%
df_data['Family'] = df_data['Parch'] + df_data['SibSp']

# %%
df_data['Sex'] = df_data['Sex'].astype('category').cat.codes
df_data['Embarked'] = df_data['Embarked'].astype('category').cat.codes

# %%
df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']

# %%
select_feat = ['Sex','Pclass','Embarked','Family']
selector = RandomForestClassifier(n_estimators=250,criterion='entropy',min_samples_split=20)
selector.fit(X[select_feat], Y)
print(selector)

# %%
submit_x = df_test.drop(labels=['PassengerId'],axis=1)

connect_pred = selector.predict(submit_x[select_feat])

submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],
                      "Survived":connect_pred.astype(int)})
submit.to_csv("submit.csv",index=False)



