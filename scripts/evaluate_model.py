import mlflow
import mlflow.pytorch
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow 서버 URI 설정

# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description='Evaluate PyTorch MNIST Model')
parser.add_argument('--model_name', type=str, default="SimpleNN", help='Name of the registered model')
parser.add_argument('--model_version', type=int, default=1, help='Version of the registered model')
args = parser.parse_args()

# 데이터셋 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='MNIST_data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 모델 로드
model = mlflow.pytorch.load_model(f"models:/{args.model_name}/{args.model_version}")

# 모델 평가
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')

print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

# MLflow에 결과 기록
with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
