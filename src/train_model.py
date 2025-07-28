import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("data.csv")
X = df.iloc[:, :-1].values  # All columns except the last one
y = df.iloc[:, -1].values  # The last column as labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print("Training complete.")
print("Model accuracy on training set: ", clf.score(X_train, y_train))
print("Model accuracy on test set: ", clf.score(X_test, y_test))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)
print("Saved in: model.pkl")