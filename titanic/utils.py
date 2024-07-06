import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataclasses import dataclass


@dataclass
class Score:
    accuracy: float
    precision: float
    recall: float
    f1_score: float


def transform_dataframe(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    df = (
        df.drop(columns=columns_to_drop, axis=1)
        .pipe(_fill_na_values)
        .pipe(_add_extra_features)
        .pipe(_label_columns)
    )
    return df

def get_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> Score:
    return Score(
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
    )

def _fill_na_values(df: pd.DataFrame) -> pd.DataFrame:
    mean_age = df["Age"].mean()
    mean_fare = df["Fare"].mean()
    mode_embarked = df["Embarked"].mode()[0]

    df = (
        df.assign()
        .pipe(lambda d: d.assign(Age=d["Age"].fillna(mean_age)))
        .pipe(lambda d: d.assign(Embarked=d["Embarked"].fillna(mode_embarked)))
        .pipe(lambda d: d.assign(Fare=d["Fare"].fillna(mean_fare)))
    )
    return df


def _add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.assign()
        .pipe(lambda d: d.assign(Family_Size=d["SibSp"] + d["Parch"]))
        .pipe(lambda d: d.assign(Fare_Per_Person=d["Fare"] / (d["Family_Size"] + 1)))
        .pipe(lambda d: d.assign(Age_Class=d["Age"] * d["Pclass"]))
    )
    return df


def _label_columns(df: pd.DataFrame) -> pd.DataFrame:
    label_encoder = LabelEncoder()
    df = (
        df.assign()
        .pipe(lambda d: d.assign(Sex=label_encoder.fit_transform(d["Sex"])))
        .pipe(lambda d: d.assign(Embarked=label_encoder.fit_transform(d["Embarked"])))
        .pipe(
            lambda d: d.assign(
                Age=pd.cut(
                    d["Age"], bins=[0, 16, float("inf")], labels=[0, 1], right=False
                )
            )
        )
    )
    return df
