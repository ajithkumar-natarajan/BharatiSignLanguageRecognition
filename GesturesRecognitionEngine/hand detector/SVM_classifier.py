from sklearn.svm import SVC
import pandas as pd

heart = pd.read_csv('centers.csv')
X = heart.iloc[0:9, 0:2]
Y = heart.iloc[0:9, 2]

clf=SVC(kernel='linear')
clf.fit(X,Y)
