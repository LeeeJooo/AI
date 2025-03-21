import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# MODEL 저장 경로
SAVE_DIR = "model"
os.makedirs(SAVE_DIR, exist_ok=True)                  # model 폴더가 없으면 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
PATH = os.path.join(SAVE_DIR, f"quartic_model_{timestamp}.pth")  # model/quadratic_model.pth

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QuarticModel(nn.Module):
    def __init__(self):
        super(QuarticModel, self).__init__()

        self.linear1 = nn.Linear(1,256)
        self.linear2 = nn.Linear(256,256)
        self.linear3 = nn.Linear(256,256)
        self.linear4 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x
    
def train(model, loss_function, optimizer, dataloader, epochs):
    print('사차 함수 학습 시작')

    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()   # 기울기 초기화
            
            # 1) Forward
            outputs = model(batch_x)

            # 2) Loss 계산
            loss = loss_function(outputs, batch_y)

            # 3) Backpropagation
            loss.backward()         # 기울기 계산
            optimizer.step()        # 가중치 업데이트 

            # 4) 미니 배치 손실 합
            epoch_loss += loss.item()
        
        # 에포크 별 손실 저장
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        # 학습 로그 출력
        if (epoch+1)%100==0:
            print(f"[EPOCH {epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    torch.save(model, PATH)
    print("사차 함수 모델 저장 완료:", PATH)

    return loss_history

def test(loss_function, inputs, targets):
    print('사차 함수 테스트 시작')

    model = torch.load(PATH)
    with torch.no_grad():       # 그래디언트 계산 비활성화
        outputs = model(inputs)  # 모델 예측값 계산
        loss = loss_function(outputs, targets)  # 손실 계산
    
    print(f"사차 함수 Test Loss: {loss.item():.6f}")

    return outputs

def visualize_results(epochs, loss_history, inputs1, targets1, predictions1, inputs2, targets2, predictions2):
    # 텐서를 NumPy 배열로 변환
    inputs1 = inputs1.cpu().numpy().flatten()
    targets1 = targets1.cpu().numpy().flatten()
    predictions1 = predictions1.cpu().numpy().flatten()
    inputs2 = inputs2.cpu().numpy().flatten()
    targets2 = targets2.cpu().numpy().flatten()
    predictions2 = predictions2.cpu().numpy().flatten()

    _, axs = plt.subplots(1, 3, figsize=(14, 6))  # 1행 3열의 그래프 생성

    # [TRAIN] 학습 손실 그래프
    axs[0].plot(range(epochs), loss_history, label="Loss", color="purple", alpha=0.5)
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("[Quartic] Training Loss")
    axs[0].set_ylim(-0.01, 1.0)
    axs[0].legend()
    axs[0].grid(True)

    # [TEST] 학습 데이터셋과 일치하는 범위
    axs[1].scatter(inputs1, targets1, label="Actual", color="blue", alpha=0.5)  # 실제 값
    axs[1].scatter(inputs1, predictions1, label="Predicted", color="red", alpha=0.5)  # 예측 값
    axs[1].set_xlabel("Input (x)")
    axs[1].set_ylabel("Output (y)")
    axs[1].set_title("[Quartic] TEST [-10,10]")
    axs[1].legend()
    axs[1].grid(True)

    # [TEST] 학습 데이터셋과 겹치지 않는 범위
    axs[2].scatter(inputs2, targets2, label="Actual", color="blue", alpha=0.5)  # 실제 값
    axs[2].scatter(inputs2, predictions2, label="Predicted", color="red", alpha=0.5)  # 예측 값
    axs[2].set_xlabel("Input (x)")
    axs[2].set_ylabel("Output (y)")
    axs[2].set_title("[Quartic] TEST [-20,-10]")
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()  # 그래프 간격 자동 조정
    plt.show()

if __name__ == "__main__":
    # 1) 하이퍼파라미터 설정
    EPOCHS = 10000
    LEARNING_RATE = 0.001
    DATASET_SIZE = 10000
    MIN_VALUE, MAX_VALUE = -10.0, 10.0
    TEST_MIN_VALUE, TEST_MAX_VALUE = -20.0, -10.0
    BATCH_SIZE = 64

    # 2) 모델 초기화
    model = QuarticModel().to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # 3) 학습 데이터 생성 : x^4
    x = torch.empty(DATASET_SIZE, dtype=torch.float32, device=device).uniform_(MIN_VALUE, MAX_VALUE).unsqueeze(1)
    y = x ** 4
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4) 학습
    loss_history = train(model, loss_function, optimizer, dataloader, EPOCHS)

    # 5) 테스트
    x_test1 = torch.empty(DATASET_SIZE, dtype=torch.float32, device=device).uniform_(MIN_VALUE, MAX_VALUE).unsqueeze(1)
    y_test1 = x_test1 ** 4

    x_test2 = torch.empty(DATASET_SIZE, dtype=torch.float32, device=device).uniform_(TEST_MIN_VALUE, TEST_MAX_VALUE).unsqueeze(1)
    y_test2 = x_test2 ** 4
    
    test_outputs1 = test(loss_function, x_test1, y_test1)
    test_outputs2 = test(loss_function, x_test2, y_test2)

    # 6) 결과 시각화
    visualize_results(EPOCHS, loss_history, x_test1, y_test1, test_outputs1, x_test2, y_test2, test_outputs2)