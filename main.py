import yaml

from widrow_hoff import Widrow
from sklearn.datasets import make_blobs
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from LogisticRegression import LogisticRegression
from sklearn.model_selection import train_test_split


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config = load_config("config.yml")
    model_name = config["model"]["name"]
    learning_rate = float(config["parameters"]["learning_rate"])
    epochs = int(config["parameters"]["epochs"])
    normalize = config["parameters"]["normalize"]


    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
    data = load_breast_cancer()

    features, labels = load_diabetes(return_X_y=True)
    labels = (labels > labels.mean()).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.1, random_state=42
    )
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
    model = None
    if model_name == "LogisticRegression":
        model = LogisticRegression(
            learning_rate, epochs, input_size=X_train.shape[1]
        )
    elif model_name == "SVM":
        print("ADD SVM MODEL HERE")
        # Add svm model here
    elif model_name == "widrow":
        model = Widrow(X_train[0].shape[0], epochs, learning_rate)
    else:
        raise ValueError(
            f"Unsupported model name passed in config file: {model_name}. Please choose from LogisticRegression, 'SVM', or 'Widrow'."
        )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, predictions)

    # Output the accuracy
    print(f"Accuracy on test dataset: {accuracy * 100:.2f}%")
