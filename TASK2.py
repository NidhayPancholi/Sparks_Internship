df=pd.read_csv("/kaggle/input/tsf-datasets/Iris.csv")
d={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
df['Species'].replace(d,inplace=True)
df.head()

df.drop('Id',axis=1,inplace=True)
X=df.drop('Species',axis=1)
y=df['Species']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
kmeans=KMeans(n_clusters=3)
kmeans.fit(X_train,y_train)

y_train_pred=kmeans.predict(X_train)
accuracy_score(y_train,y_train_pred)

y_pred=kmeans.predict(X_test)
accuracy_score(y_test,y_pred)

sns.scatterplot(data=X_train,x='SepalLengthCm',y='SepalWidthCm',hue=y_train)

sns.scatterplot(data=X_train,x='SepalLengthCm',y='SepalWidthCm',hue=y_train_pred)

sns.scatterplot(data=X_train,x='PetalLengthCm',y='PetalWidthCm',hue=y_train)
sns.scatterplot(data=X_train,x='PetalLengthCm',y='PetalWidthCm',hue=y_train_pred)
