import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
class myLinearRegression:
    @staticmethod
    def labelEncoder(df,column):
        df[column].unique().tolist()
        for i,j in enumerate(df[column].unique().tolist()):
            print(i,j)
            df.replace(j,i,inplace=True)
        return df
    @staticmethod
    def split(X,Y):
        total_data=len(X)
        train_data=total_data*70/100
        test_data=total_data*30/100
        print(total_data,train_data,test_data)
        train_data_X=X.head(int(train_data))
        test_data_X=X.tail(int(test_data))
        train_data_Y=Y.head(int(train_data))
        test_data_Y=Y.tail(int(test_data))
        return [train_data_X,test_data_X,train_data_Y,test_data_Y]
    def fit(self,X,Y):
        np.set_printoptions(suppress=True)
        X=X.to_numpy()
        Y=Y.to_numpy()
        X=np.column_stack((np.ones(len(X)),X))
        M=np.linalg.inv(X.T@X)@X.T@Y
        #or
        # M=np.linalg.pinv(X)@Y
        self.intercept_=M[0]
        self.coeff_=M[1:]
    def predict(self,test):
        R=[]
        p=self.intercept_
        for j in test:
            for i,c in enumerate(self.coeff_):
                p=p+c*j[i]
            R.append(p)
        return R
df=pd.read_csv("housing_data.csv")
df=df.dropna() 
df=myLinearRegression.labelEncoder(df,'ocean_proximity')
Y=df["median_house_value"]
X=df.drop("median_house_value", axis=1)
train_data_X,test_data_X,train_data_Y,test_data_Y=myLinearRegression.split(X,Y)
model=myLinearRegression()
model.fit(train_data_X,train_data_Y)
print(model.intercept_)
print(model.coeff_)
T=test_data_X.to_numpy()
p=model.predict(T)
print(p)
'''________________________________________________________________________________________________________'''
Y=df["median_house_value"]
X_simple=df[["median_income"]]
model=LinearRegression()
model.fit(X_simple,Y)
plt.scatter(X_simple,Y)
plt.plot(X_simple,model.predict(X_simple),color='green')
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Median Income vs Median House Value")
plt.show()