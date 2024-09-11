from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

pipelines = [
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("clf", LogisticRegression(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline([("scaler", StandardScaler()), ("clf", SVC(class_weight="balanced"))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", SVC(class_weight="balanced"))]),
    Pipeline([("scaler", RobustScaler()), ("clf", SVC(class_weight="balanced"))]),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(class_weight="balanced")),
        ]
    ),
    Pipeline(
        [("scaler", MinMaxScaler()), ("clf", RidgeClassifier(class_weight="balanced"))]
    ),
    Pipeline(
        [("scaler", RobustScaler()), ("clf", RidgeClassifier(class_weight="balanced"))]
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("clf", DecisionTreeClassifier(class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("scaler", RobustScaler()),
            ("clf", DecisionTreeClassifier(class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("clf", RandomForestClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("scaler", RobustScaler()),
            ("clf", RandomForestClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("clf", ExtraTreesClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("scaler", RobustScaler()),
            ("clf", ExtraTreesClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline([("scaler", StandardScaler()), ("clf", SGDClassifier(n_jobs=-1))]),
    Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("clf", SGDClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("scaler", RobustScaler()),
            ("clf", SGDClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    )
]


def get_pipelines():
    return pipelines