import yaml

from widrow_hoff import Widrow
from sklearn.datasets import load_diabetes, make_blobs
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
    dataset = config["dataset"]["name"]

    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
    data = None
    if dataset == "cancer":
        data = load_breast_cancer(return_X_y=True)
    elif dataset == "generate":
        data = make_blobs(n_samples=1000, n_features=1, centers=2 ,random_state=0)
    else:
        raise ValueError(
            f"Unsupported dataset name passed in config file: {dataset}. Please choose from 'cancer', or 'generate'."
        )

    features, labels = data
    # labels = (labels > labels.mean()).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.1, random_state=4
    )
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
    model = None
    if model_name == "LogisticRegression":
        model = LogisticRegression(learning_rate, epochs, input_size=X_train.shape[1])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, predictions)

        # Output the accuracy
        print(f"Accuracy on test dataset: {accuracy * 100:.2f}%")
    elif model_name == "SVM":
        print("ADD SVM MODEL HERE")
        # Add svm model here
    elif model_name == "widrow":
        model = Widrow(X_train[0].shape[0])
        model.fit(X_train, y_train, epochs, lr=learning_rate)
        predictions = model.forward(X_test)

        accuracy = accuracy_score(y_test, predictions)

        # Output the accuracy
        print(f"Accuracy on test dataset: {accuracy * 100:.2f}%")
    else:
        raise ValueError(
            f"Unsupported model name passed in config file: {model_name}. Please choose from LogisticRegression, 'SVM', or 'Widrow'."
        )
