from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import random

parkinsons = fetch_ucirepo(id=174)

#Are imported as dfs
X = parkinsons.data.features
y = parkinsons.data.targets


def EDA(features, targets):
    print(f'{features.shape[1]} Features')
    print(features.info())
    print(features.describe())
    print(targets.value_counts())

    corr = features.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
    features.hist(figsize=(15, 15))
    plt.show()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c='cyan')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Parkinson's Dataset")
    plt.show()

#EDA(X, y)
#for git, make one repo for eda, one for data describing, one for models
X = X.to_numpy()
y = y.to_numpy()
y = y.ravel()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from UCI_neuralnet import UCIClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = UCIClassifier().to(device)
net.load_state_dict(torch.load('best_model_UCI.pt'))

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),
    'Neural Network': net
}

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

model_f1s = []

for name, model in models.items():
    if isinstance(model, BaseEstimator):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        model_f1s.append(f1_score(y_test,preds))

    elif isinstance(model, nn.Module):
        model.eval()
        y_true = []
        y_pred_list = []
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long().squeeze()
        with torch.no_grad():
            y_true.extend(y_test)
            pred = net(X_test).cpu()
            pred_probs = F.sigmoid(pred)
            y_pred = torch.argmax(pred, dim=1)
            y_pred_list.extend(y_pred.numpy())
            model_f1s.append(f1_score(y_true,y_pred_list))
    else:
        print("Model type unknown")
        break

    print(f"{name}:")
    if isinstance(model, BaseEstimator):
        print(f"  Accuracy: {accuracy_score(y_test, preds)}")
        print(f"  F1 Score: {f1_score(y_test, preds)}")
        print(f"  AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")
    else:
        print(f"  Accuracy: {accuracy_score(y_true,y_pred_list)}")
        print(f"  F1 Score: {f1_score(y_true,y_pred_list)}")
        y_probs = pred_probs.cpu().detach().numpy()
        print(f"  AUC: {roc_auc_score(y_true, y_probs[:, 1])}")

# has_duplicates = len(model_f1s) != len(set(model_f1s))

max_f_idx = np.argmax(model_f1s)
print(f"Model with highest F1 score: {list(models)[max_f_idx]} at {round(max(model_f1s),4)}")
best_model = list(models.values())[max_f_idx]

print(f'Predictions with {list(models)[max_f_idx]}')
idx = random.randint(0, len(X_test-9))
data_to_pred = X_test[idx:idx+9]

if isinstance(best_model, BaseEstimator):
    outputs = best_model.predict(data_to_pred)
else:
    outputs = best_model(data_to_pred)
    outputs = outputs.argmax(dim=1).numpy()

print(f"Predicted classes: {outputs}")
print(f"True labels: {y_test[idx:idx+9].numpy()}")