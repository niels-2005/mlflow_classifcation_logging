from sklearn import metrics
from pipelines import get_pipelines
from load_dataset import get_data_usable
from evaluate_pipelines import plot_classification_report_with_support, make_confusion_matrix
import mlflow
import os

# mlflow ui 
# python compare_pipelines.py


# tracking Url und Experiment Name
mlflow.set_tracking_uri("http://0.0.0.0:5000/")
mlflow.set_experiment("Compare Pipelines Test with PNG 2")


def mlflow_logging():

    # get pipelines
    pipelines = get_pipelines()
    # get data
    X_train, X_test, y_train, y_test = get_data_usable()
    # define classes
    classes = ["No Diabetes", "Diabetes"]

    for i, pipe in enumerate(pipelines):
        print(f"Pipe: {i+1} from {len(pipelines)}")

        # start mlflow run
        with mlflow.start_run() as run:

            # fit pipe and predict
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            # log parameters from pipe with named_steps
            mlflow.log_params(pipe.named_steps)

            # get classifier and scaler name for correctfolder
            clf_name = pipe.named_steps["clf"].__class__.__name__
            scaler_name = pipe.named_steps["scaler"].__class__.__name__
            folder_path = f"pipeline_metrics/{clf_name + scaler_name}"


            # if folder_path not exists, create one
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # generate classification_report
            report = metrics.classification_report(
                    y_pred=y_pred, y_true=y_test, output_dict=True, target_names=classes
                )
            # generate classification report as plot and confusion matrix
            plot_classification_report_with_support(report=report, folder_path=folder_path)
            make_confusion_matrix(y_pred=y_pred, y_true=y_test, classes=classes, folder_path=folder_path)


            # log plots as artifacts
            mlflow.log_artifact(f"{folder_path}/classification_report.png")
            mlflow.log_artifact(f"{folder_path}/confusion_matrix.png")
            

            # calculate basic metrics
            accuracy = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)
            f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred, average="weighted")
            precision = metrics.precision_score(y_true=y_test, y_pred=y_pred, average="weighted")
            recall = metrics.recall_score(y_true=y_test, y_pred=y_pred, average="weighted")
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("F1-Score", f1)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            

# cmd
if __name__ == "__main__":
    mlflow_logging()




