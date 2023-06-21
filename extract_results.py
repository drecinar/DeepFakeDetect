import argparse
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt


def extract_results(true_csv_path, predicted_csv_path):
    df_true = pd.read_csv(true_csv_path)
    df_predict = pd.read_csv(predicted_csv_path)

    report = metrics.classification_report(df_true['label'], df_predict['label'])
    print(report)
    report = metrics.accuracy_score(df_true['label'], df_predict['label'])
    print("Accuracy: ", report)
    report = metrics.precision_score(df_true['label'], df_predict['label'])
    print("Precision: ", report)
    report = metrics.recall_score(df_true['label'], df_predict['label'])
    print("Recall: ", report)
    conf = metrics.confusion_matrix(df_true['label'], df_predict['label'])
    print(conf)
    auc = metrics.roc_auc_score(df_true['label'], df_predict['label'])
    print("AUC: ", auc)

    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf,
                                          display_labels=["Real", "Fake"])
    disp.plot()
    plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True,
                        help='Path of submission.csv which generated from model')

    opt = parser.parse_args()
    extract_results("./data/labels.csv", opt.path)



