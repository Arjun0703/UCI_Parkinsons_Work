import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parkinsons = fetch_ucirepo(id=174)
X = parkinsons.data.features
y = parkinsons.data.targets

X = X.to_numpy()
y = y.to_numpy()
y = y.ravel()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long().squeeze()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)
X_train = X_train.to(device)
X_val = X_val.to(device)
y_train = y_train.to(device)
y_val = y_val.to(device)


class UCIClassifier(nn.Module):
    def __init__(self, num_units=32, dropout=0.3):
        super(UCIClassifier, self).__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(22, num_units),
            nn.ReLU(),
            nn.Linear(num_units, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
    def forward(self, x):
        return self.fc_block(x)

net = UCIClassifier().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

def train(net,X,y):
    optimizer.zero_grad()
    prediction = net(X)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()
    return loss.data

def validate(net,X,y):
    with torch.no_grad():
        prediction = net(X)
        loss = loss_fn(prediction, y)
        return loss

train_losses=[]
val_losses=[]
epochs = 1000
for i in range(epochs):
    loss = train(net, X_train, y_train)
    train_losses.append(loss) # we save losses to display them at the end
    val_losses.append(validate(net,X_val,y_val))

    if i>100:
        if val_losses[i-1]>val_losses[i-2]>val_losses[i-3]:
            best_model_state = net.state_dict()
            print(f"Final iteration: {i}")
            break

if best_model_state:
    net.load_state_dict(best_model_state)
    torch.save(net.state_dict(), 'best_model_UCI.pt')

from sklearn.metrics import accuracy_score, f1_score

def accuracy_f1(net, features, labels, text='Evaluation: '):
    net.eval()
    y_true = []
    y_pred_list = []
    with torch.no_grad():
        y_true.extend(labels.cpu().numpy())
        pred = net(features.cpu()).cpu()
        y_pred = torch.argmax(pred, dim=1)
        y_pred_list.extend(y_pred.numpy())

    accuracy = accuracy_score(y_true, y_pred_list)
    f1 = f1_score(y_true, y_pred_list)

    print(f"{text} Accuracy: {accuracy:.2f}, F1-score: {f1:.2f}")
    return accuracy, f1


accuracy_f1(net, X_test, y_test, 'Test accuracy: ')

idx = random.randint(0, len(X_test-9))
data_to_pred = X_test[idx:idx+9].to(device)
with torch.no_grad():
    output = net(data_to_pred)

pred_class = output.argmax(dim=1)
print(f"Predicted classes: {pred_class}")
print(f"True labels: {y_test[idx:idx+9]}")
