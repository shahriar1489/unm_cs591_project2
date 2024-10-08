import yaml

from widrow_hoff import Widrow
from svm_binary import SVM
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from LogisticRegression import LogisticRegression
from weston_watkins import Weston_Watkins
from sklearn.model_selection import train_test_split
import time


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config = load_config("config.yml")
    model_name = config["model"]["name"]
    lr = float(config["parameters"]["learning_rate"])
    epochs = int(config["parameters"]["epochs"])
    normalize = config["parameters"]["normalize"]
    dataset = config["dataset"]["name"]

    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
    data = None
    if dataset == "cancer":
        data = load_breast_cancer(return_X_y=True)
    elif dataset == "generate":
        data = make_blobs(n_samples=1000, n_features=1, centers=2, random_state=0)
    elif dataset =='iris':
        data = load_iris(return_X_y=True)
    else:
        raise ValueError(
            f"Unsupported dataset name passed in config file: {dataset}. Please choose from 'cancer', 'iris(multi)'or 'generate'."
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


    start_time = time.time()
    if model_name == "weston":
        observations,features = X_train.shape 
        model =  Weston_Watkins(features,lr,epochs)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Output the accuracy
        print(f"Accuracy on test {dataset} dataset: {accuracy * 100:.2f}%")
        
    elif model_name == "LogisticRegression":
        model = LogisticRegression(lr, epochs, input_size=X_train.shape[1])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, predictions)

        # Output the accuracy
        print(f"Accuracy on test dataset: {accuracy * 100:.2f}%")
    elif model_name == "SVM":

        # Initialize the SVM model
        model = SVM(learning_rate=0.01, n_iters=1000)

        # Fit the model on the dataset
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, predictions)

        # Output the accuracy
        print(f"Accuracy on test dataset: {accuracy * 100:.2f}%")

    elif model_name == "widrow":
        model = Widrow(X_train[0].shape[0])
        model.fit(X_train, y_train, epochs, lr=lr)
        predictions = model.forward(X_test)

        accuracy = accuracy_score(y_test, predictions)

        # Output the accuracy
        print(f"Accuracy on test dataset: {accuracy * 100:.2f}%")
    else:
        raise ValueError(
            f"Unsupported model name passed in config file: {model_name}. Please choose from 'LogisticRegression', 'SVM','weston' or 'Widrow'."
        )

end_time = time.time()
print(f"Time taken for {model_name}: {end_time - start_time:.2f} seconds")
