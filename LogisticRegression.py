import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
class LogisticRegression:
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
            K=1/(1+np.exp(p))
            R.append(K)
        return R
df=pd.read_csv("testing.csv")
# print(df.head(5))
df=LogisticRegression.labelEncoder(df,"prognosis")
# print(df.head(5)
X=df[["itching","skin_rash","continuous_sneezing","shivering","stomach_pain","acidity","vomiting","spotting_ urination"]]
Y=df["prognosis"]
train_data_X,test_data_X,train_data_Y,test_data_Y=LogisticRegression.split(X,Y)
# print(test_data_X)
model=LogisticRegression()
model.fit(X,Y)
print(model.intercept_)
print(model.coeff_)
p=model.predict(test_data_X.to_numpy())
print(p)