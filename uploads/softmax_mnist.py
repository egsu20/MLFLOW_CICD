import torch #PyTorch의 핵심 패키지로, 다차원 텐서(다차원 배열 통칭)를 지원하는 머신 러닝 프레임워크
import torchvision.datasets as dsets # 이미지 데이터셋을 다루는 데 사용되는 PyTorch 패키지
import torchvision.transforms as transforms # 이미지 데이터 전처리를 위한 도구를 제공
from torch.utils.data import DataLoader # PyTorch의 데이터셋을 미니 배치로 나누어 반복할 수 있는 데이터 로더. 머신 러닝 모델을 학습할 때 사용
import torch.nn as nn # PyTorch의 신경망 모델을 구축하는 데 사용되는 패키지
import matplotlib.pyplot as plt # 데이터 시각화를 위한 Python 라이브러리
import random # 난수 생성 및 관련된 기능을 제공하는 Python 표준 라이브러리

print("update")

USE_CUDA = torch.cuda.is_available() # GPU를 사용할 수 있으면 True, 아니면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU를 사용할 수 있으면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:",device)

# for reproducibility(재현성)
random.seed(777) # random의 시드를 777로 설정. 이렇게 하면 코드가 실행될 때마다 동일한 순서로 난수가 생성된다.
torch.manual_seed(777) # PyTorch의 난수 발생기에 대한 시드를 설정
if device == 'cuda': # CUDA 연산을 사용하는 경우
  torch.cuda.manual_seed_all(777) # GPU의 난수 발생기에 대한 시드를 설정. 이렇게 하면 GPU를 사용할 때도 동일한 결과를 얻을 수 있음.

#hyperparameters
training_epochs = 15
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/', # MNIST 데이터를 다운로드 받을 경로
                          train=True, # 훈련 데이터를 리턴받음
                          transform=transforms.ToTensor(), # 현재 데이터를 파이토치 텐서로 변환해 줌.
                          download=True) # 해당 경로에 MNIST 데이터가 없다면 다운로드 받겠다는 의미

mnist_test = dsets.MNIST(root='MNIST_data/', # MNIST 데이터를 다운로드 받을 경로
                         train=False, # 테스트 데이터를 리턴받음
                         transform=transforms.ToTensor(), # 현재 데이터를 파이토치 텐서로 변환해 줌.
                         download=True) # 해당 경로에 MNIST 데이터가 없다면 다운로드 받겠다는 의미

# dataset loader
data_loader = DataLoader(dataset=mnist_train, # 로드할 대상
                                          batch_size=batch_size, # 배치 크기는 100
                                          shuffle=True, # 매 에포크마다 미니 배치를 셔플할 것인지의 여부
                                          drop_last=True) # 마지막 배치를 버릴 것인지

# MNIST data image of shape 28 * 28 = 784
linear = nn.Linear(784, 10, bias=True).to(device) # input_dim = 784, output_dim = 10

# 비용 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device) # 내부적으로 소프트맥스 함수를 포함하고 있음.
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1) # SGD(Stochastic Gradient Descent)를 사용하여 모델의 학습 가능한 파라미터를 업데이트

for epoch in range(training_epochs):  # training_epochs: 전체 데이터셋을 몇 번 반복할 것인지를 결정하는 에포크 수
    avg_cost = 0  # 각 에포크마다 손실의 평균을 저장할 변수를 초기화
    total_batch = len(data_loader) # 전체 배치의 개수

    for X, Y in data_loader:  # 데이터로더를 통해 미니 배치(전체 데이터셋을 작은 일부 그룹으로 나누는 것) 단위로 데이터를 가져옴
        # 입력 데이터를 2차원에서 1차원으로 평탄화한다. MNIST 데이터의 경우 28x28 크기이므로 784로 변환된다.
        X = X.view(-1, 28 * 28).to(device)
        # 레이블을 장치에 올림. (CPU나 GPU 같은 장치로 이동시키는 것)
        Y = Y.to(device)

        optimizer.zero_grad()  # 기울기를 초기화합니다.
        hypothesis = linear(X)  # 모델에 입력을 넣고 예측값을 계산합니다.
        cost = criterion(hypothesis, Y)  # 비용 함수를 사용하여 손실을 계산합니다.
        cost.backward()  # 역전파를 통해 기울기를 계산합니다.
        optimizer.step()  # 옵티마이저를 사용하여 모델 파라미터를 업데이트합니다.

        avg_cost += cost / total_batch  # 배치 손실을 전체 배치 개수로 나눠 평균 손실을 계산합니다.

    # 각 에포크가 끝날 때마다 현재 에포크의 번호와 평균 손실을 출력합니다.
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

# 학습이 완료되었음을 출력합니다.
print('Learning finished')


# 테스트 데이터를 사용하여 모델을 평가합니다.
with torch.no_grad():  # torch.no_grad()를 호출하면 gradient 계산을 수행하지 않습니다.
    # 테스트 데이터를 준비합니다. 데이터를 평탄화하고 장치로 이동합니다.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    # 모델을 사용하여 예측을 수행합니다.
    prediction = linear(X_test)
    # 예측한 클래스 중 가장 큰 값을 선택하여 정확도를 계산합니다.
    correct_prediction = torch.argmax(prediction, 1) == Y_test # 모델이 테스트 데이터셋을 통해 예측한 클래스에 대한 확률 분포
    accuracy = correct_prediction.float().mean()  # 정확한 예측의 비율을 계산합니다.
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 선택하여 해당 이미지의 예측을 출력합니다.
    r = random.randint(0, len(mnist_test) - 1)  # 무작위로 인덱스를 선택합니다.
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label:', Y_single_data.item())  # 선택된 이미지의 실제 레이블을 출력합니다.
    single_prediction = linear(X_single_data)  # 모델을 사용하여 예측을 수행합니다.
    print('Prediction:', torch.argmax(single_prediction, 1).item())  # 예측된 클래스를 출력합니다.

    # 선택된 이미지를 시각화하여 출력합니다.
    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

