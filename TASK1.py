from sklearn.linear_model import LinearRegression 
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error,accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

df=pd.read_csv("/kaggle/input/tsf-datasets/student_scores.csv")
df.head()

def fit():
    x=np.array(df['Hours']).reshape(-1,1)
    y=df['Scores']
    linear=LinearRegression()
    linear.fit(x,y)
    print('The mean absolute error is :',mean_absolute_error(linear.predict(x),y))
    return linear
def pred(val,model):
    val=np.array([val]).reshape(-1,1)
    return model.predict(val)[0]
    
    model=fit()
    
    pred(9.25,model)
    
h=[x for x in range(0,11)]
h=np.array(h)
p=model.coef_[0]*h+model.intercept_
sns.lineplot(x=h,y=p,label='model',color='red')
sns.scatterplot(x=df['Hours'],y=df['Scores'])
