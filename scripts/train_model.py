import mlflow
import mlflow.pytorch
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mlflow.tracking import MlflowClient
import numpy as np  # numpy import 추가
from sklearn.metrics import precision_score, recall_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow 서버 URI 설정

# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate (default: 0.01)')
args = parser.parse_args()

# 하이퍼파라미터 설정
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

# 데이터셋 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='MNIST_data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='MNIST_data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# MLflow 시작
experiment_name = "MNIST Experiment"
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    client.create_experiment(experiment_name)
else:
    if experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)

mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)

    # 모델 학습
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
        avg_loss = running_loss / len(train_loader)
        epoch_accuracy = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
        epoch_precision = precision_score(all_labels, all_preds, average='macro')
        epoch_recall = recall_score(all_labels, all_preds, average='macro')
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}')
        mlflow.log_metric("loss", avg_loss, step=epoch)
        mlflow.log_metric("accuracy", epoch_accuracy, step=epoch)
        mlflow.log_metric("precision", epoch_precision, step=epoch)
        mlflow.log_metric("recall", epoch_recall, step=epoch)

    # 모델 저장
    mlflow.pytorch.log_model(model, "model")
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "SimpleNN")
