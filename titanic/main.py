import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

training_titanic_df = pd.read_csv("./titanic/data/train.csv")

columns_to_drop = ["Ticket", "Cabin", "Name", "PassengerId"]

y_column = "Survived"
y = training_titanic_df[y_column]
x = transform_dataframe(training_titanic_df, columns_to_drop + [y_column])

x_train, x_cross_validation, y_train, y_cross_validation = train_test_split(
    x, y, test_size=0.15, random_state=42
)

pipeline = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression())])

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_cross_validation)
score = get_classification_metrics(y_cross_validation, y_pred)

print(score)

test_titanic_df = pd.read_csv("./titanic/data/test.csv")

xtest = transform_dataframe(test_titanic_df, columns_to_drop)
ytest = pipeline.predict(xtest)

passengers_id = test_titanic_df["PassengerId"]
result = pd.DataFrame({"PassengerId": passengers_id, "Survived": ytest})

result.to_csv("./titanic/data/output.csv", index=False)
