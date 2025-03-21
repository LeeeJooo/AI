import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QuadraticModel(nn.Module):
    def __init__(self):
        super(QuadraticModel, self).__init__()

        self.linear1 = nn.Linear(1,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,64)

        self.out1 = nn.Linear(64,10) # 백의 자리
        self.out2 = nn.Linear(64,10) # 십의 자리
        self.out3 = nn.Linear(64,10) # 일의 자리

        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))

        out1 = self.log_softmax(self.out1(x))
        out2 = self.log_softmax(self.out2(x))
        out3 = self.log_softmax(self.out3(x))
        
        return out1, out2, out3

def predict(out1, out2, out3):
    pred1 = torch.argmax(out1, dim=1)
    pred2 = torch.argmax(out2, dim=1)
    pred3 = torch.argmax(out3, dim=1)
    return torch.stack((pred1, pred2, pred3), dim=1)  # (Batch_Size, 3)

def evaluate(model, loss_function, inputs_eval_batches, labels_eval_batches):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs_batch, labels_batch in zip(inputs_eval_batches, labels_eval_batches):
            out1, out2, out3 = model(inputs_batch)

            # 개별 손실 계산
            loss1 = loss_function(out1, torch.argmax(labels_batch, dim=2)[:, 0])
            loss2 = loss_function(out2, torch.argmax(labels_batch, dim=2)[:, 1])
            loss3 = loss_function(out3, torch.argmax(labels_batch, dim=2)[:, 2])
            loss = loss1 + loss2 + loss3

            total_loss += loss.item()

            # 예측값 가져오기
            pred1, pred2, pred3 = torch.argmax(out1, dim=1), torch.argmax(out2, dim=1), torch.argmax(out3, dim=1)
            predictions = torch.stack((pred1, pred2, pred3), dim=1)

            # 정확도 계산
            labels = torch.argmax(labels_batch, dim=2)
            correct_predictions += (predictions == labels).all(dim=1).sum().item()
            total_samples += labels.shape[0]

    avg_loss = total_loss / len(inputs_eval_batches)
    accuracy = (correct_predictions / total_samples) * 100

    model.train()

    return avg_loss, accuracy

def train(model, loss_function, optimizer, inputs_batches, labels_batches, inputs_eval, labels_eval, epochs, eval_interval):
    loss_history, acc_history = [], []
    eval_loss_history, eval_acc_history = [], []

    for epoch in range(epochs):

        total_loss = 0.0  # 한 epoch의 총 손실
        correct_predictions = 0
        total_samples = 0

        for inputs_batch, labels_batch in zip(inputs_batches, labels_batches):
            out1, out2, out3 = model(inputs_batch)

            # out (batch_size, 10)
            # labels_batch (batch_size, 3, 10)
            labels = torch.argmax(labels_batch, dim=2).to(torch.long)
            loss1 = loss_function(out1, torch.argmax(labels_batch, dim=2)[:, 0])
            loss2 = loss_function(out2, torch.argmax(labels_batch, dim=2)[:, 1])
            loss3 = loss_function(out3, torch.argmax(labels_batch, dim=2)[:, 2])

            loss = loss1 + loss2 + loss3  # 총 손실 계산
            optimizer.zero_grad()
            loss.backward()  # Backpropagation (기울기 계산)
            optimizer.step()  # 가중치 업데이트

            total_loss += loss.item()

            # 정확도 계산
            predictions = predict(out1, out2, out3)
            labels = torch.argmax(labels_batch, dim=2)
            correct_predictions += (predictions == labels).all(dim=1).sum().item()
            total_samples += labels_batch.shape[0]

        # Epoch 단위 평균 손실
        train_loss = total_loss / len(labels_batches)
        loss_history.append(train_loss)

        # Epoch 단위 정확도
        train_acc = (correct_predictions / total_samples) * 100
        acc_history.append(train_acc)

        if (epoch+1) % eval_interval == 0:
            eval_loss, eval_acc = evaluate(model, loss_function, inputs_eval, labels_eval)
            eval_loss_history.append(eval_loss)
            eval_acc_history.append(eval_acc)
            print(f"[Epoch {epoch+1:4.0f}/{epochs}]  Train Loss: {train_loss:.4f}    |   Train Acc: {train_acc:.2f}  |   Eval Loss: {eval_loss:.4f}  |   Eval Acc: {eval_acc:.2f}")

    return loss_history, acc_history, eval_loss_history, eval_acc_history

def preprocess(dataset):
    dataset_size = dataset.shape[0]
    hundreds = torch.div(dataset, 100, rounding_mode='floor').to(torch.int64)  # 백의 자리
    tens = torch.div(dataset % 100, 10, rounding_mode='floor').to(torch.int64)  # 십의 자리
    ones = (dataset % 10).to(torch.int64)   # 일의 자리

    processed_dataset = torch.zeros((dataset_size, 3, 10), dtype=torch.int64, device=device)
    processed_dataset[:, 0, :].scatter_(1, hundreds, 1)  # 백의 자리
    processed_dataset[:, 1, :].scatter_(1, tens, 1)  # 십의 자리
    processed_dataset[:, 2, :].scatter_(1, ones, 1)  # 일의 자리

    return processed_dataset.to(torch.float32)

def test(model, inputs):
    model.eval()
    with torch.no_grad():
        out1, out2, out3 = model(inputs)
        predictions = predict(out1, out2, out3)
    return predictions

def visualize_results(epochs, loss_history, acc_history, validation_epochs, eval_loss_history, eval_acc_history, inter_inputs, inter_labels, inter_predictions, inter_accuracy, extra_inputs, extra_labels, extra_predictions, extra_accuracy):
    inter_inputs = inter_inputs.cpu().numpy().flatten()
    extra_inputs = extra_inputs.cpu().numpy().flatten()
    
    inter_labels = predict(inter_labels[:,0], inter_labels[:,1], inter_labels[:,2])
    inter_labels = inter_labels[:, 0] * 100 + inter_labels[:, 1] * 10 + inter_labels[:, 2]
    inter_labels = inter_labels.cpu().numpy().flatten()

    inter_predictions = inter_predictions[:, 0] * 100 + inter_predictions[:, 1] * 10 + inter_predictions[:, 2]
    inter_predictions = inter_predictions.cpu().numpy().flatten()


    extra_labels = predict(extra_labels[:,0], extra_labels[:,1], extra_labels[:,2])
    extra_labels = extra_labels[:, 0] * 100 + extra_labels[:, 1] * 10 + extra_labels[:, 2]
    extra_labels = extra_labels.cpu().numpy().flatten()

    extra_predictions = extra_predictions[:, 0] * 100 + extra_predictions[:, 1] * 10 + extra_predictions[:, 2]
    extra_predictions = extra_predictions.cpu().numpy().flatten()

    _, axs = plt.subplots(1, 4, figsize=(20, 8))

    # TRAIN LOSS, ACCURACY 그래프
    ax_train_loss = axs[0]
    axs[0].set_title("TRAIN LOSS ACC")
    ax_train_acc = ax_train_loss.twinx()
    
    ax_train_loss.plot(range(epochs), loss_history, label="Loss", color="purple", linewidth=3)
    ax_train_loss.set_xlabel("Epochs")
    ax_train_loss.set_ylabel("Loss")
    ax_train_loss.tick_params(axis="y", labelcolor="purple")
    ax_train_loss.grid(True)

    ax_train_acc.plot(range(epochs), acc_history, label="Accuracy", color="green", linewidth=3)
    ax_train_acc.set_ylabel("Accuracy", color="green")
    ax_train_acc.tick_params(axis="y", labelcolor="green")

    ax_train_loss.legend(loc="upper left")
    ax_train_acc.legend(loc="upper right")
    
    # VALIDATION LOSS, ACCURACY 그래프
    ax_eval_loss = axs[1]
    axs[1].set_title("EVAL LOSS ACC")
    ax_eval_acc = ax_eval_loss.twinx()
    
    ax_eval_loss.plot(validation_epochs, eval_loss_history, label="Loss", color="purple", linewidth=3)
    ax_eval_loss.set_xlabel("Epochs")
    ax_eval_loss.set_ylabel("Loss")
    ax_eval_loss.tick_params(axis="y", labelcolor="purple")
    ax_eval_loss.grid(True)

    ax_eval_acc.plot(validation_epochs, eval_acc_history, label="Accuracy", color="green", linewidth=3)
    ax_eval_acc.set_ylabel("Accuracy", color="green")
    ax_eval_acc.tick_params(axis="y", labelcolor="green")

    ax_eval_loss.legend(loc="upper left")
    ax_eval_acc.legend(loc="upper right")

    inter_acc_text = f"Accuracy: {inter_accuracy:.2f}%"
    axs[2].text(
        x=-5, y=80,
        s=inter_acc_text,
        fontsize=12,
        color="black",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="black")
    )
    axs[2].scatter(inter_inputs, inter_labels, label="Actual", color="blue", alpha=0.5)
    axs[2].scatter(inter_inputs, inter_predictions, label="Predicted", color="red", alpha=0.5, s=4)
    axs[2].set_xlabel("Input (x)")
    axs[2].set_ylabel("Output (y)")
    axs[2].set_title("[INTERPOLATION] [-10, 10]")
    axs[2].legend()
    axs[2].grid(True)


    extra_acc_text = f"Accuracy: {extra_accuracy:.2f}%"
    axs[3].text(
        x=-5, y=80,
        s=extra_acc_text,
        fontsize=12,
        color="black",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="black")
    )
    axs[3].scatter(extra_inputs, extra_labels, label="Actual", color="blue", alpha=0.5)
    axs[3].scatter(extra_inputs, extra_predictions, label="Predicted", color="red", alpha=0.5, s=4)
    axs[3].set_xlabel("Input (x)")
    axs[3].set_ylabel("Output (y)")
    axs[3].set_title("[EXTRAPOLATION] [-20, -10]")
    axs[3].legend()
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    EPOCHS = 10000
    BATCH_SIZE = 16
    DATASET_SIZE = 10000
    TEST_DATASET_SIZE = 1000
    MIN_VALUE, MAX_VALUE = -10.0, 10.0
    EXTRA_MIN_VALUE, EXTRA_MAX_VALUE = -20.0, -10.0
    TRAIN_RATIO = 0.8
    EVAL_INTERVAL = 10
    LEARNING_RATE = 0.0001

    # 데이터 생성
    inputs = MIN_VALUE + (MAX_VALUE-MIN_VALUE) * torch.rand(DATASET_SIZE, dtype=torch.float32).view(DATASET_SIZE, 1).to(device)
    labels = inputs ** 2
    labels = preprocess(labels)

    # 데이터 분할
    train_size = int(DATASET_SIZE * TRAIN_RATIO)
    inputs_train, inputs_eval = inputs[:train_size], inputs[train_size:]
    labels_train, labels_eval = labels[:train_size], labels[train_size:]

    # Mini-Batch 생성
    inputs_train_batches = torch.split(inputs_train, BATCH_SIZE)
    labels_train_batches = torch.split(labels_train, BATCH_SIZE)

    inputs_eval_batches = torch.split(inputs_eval, BATCH_SIZE)
    labels_eval_batches = torch.split(labels_eval, BATCH_SIZE)

    # 모델 초기화 및 설계
    model = QuadraticModel().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history, acc_history, eval_loss_history, eval_acc_history = train(model, loss_function, optimizer, inputs_train_batches, labels_train_batches, inputs_eval_batches, labels_eval_batches, EPOCHS, EVAL_INTERVAL)

    # 테스트 : INTERPOLATION
    inter_inputs = MIN_VALUE + (MAX_VALUE-MIN_VALUE) * torch.rand(TEST_DATASET_SIZE, dtype=torch.float32).view(TEST_DATASET_SIZE, 1).to(device)
    inter_labels = inter_inputs ** 2
    inter_labels = preprocess(inter_labels)
    inter_predictions = test(model, inter_inputs)
    inter_labels_ = torch.argmax(inter_labels, dim=2)
    correct_inter_predictions = (inter_predictions == inter_labels_).all(dim=1).sum().item() / TEST_DATASET_SIZE * 100

    # 테스트 : EXTRAPOLATION
    extra_inputs = MIN_VALUE + (EXTRA_MAX_VALUE-EXTRA_MIN_VALUE) * torch.rand(TEST_DATASET_SIZE, dtype=torch.float32).view(TEST_DATASET_SIZE, 1).to(device)
    extra_labels = extra_inputs ** 2
    extra_labels = preprocess(extra_labels)
    extra_predictions = test(model, extra_inputs)
    extra_labels_ = torch.argmax(extra_labels, dim=2)
    correct_extra_predictions = (extra_predictions == extra_labels_).all(dim=1).sum().item() / TEST_DATASET_SIZE * 100

    validation_epochs = list(range(0, EPOCHS, EVAL_INTERVAL))
    visualize_results(EPOCHS, loss_history, acc_history, validation_epochs, eval_loss_history, eval_acc_history, inter_inputs, inter_labels, inter_predictions, correct_inter_predictions, extra_inputs, extra_labels, extra_predictions, correct_extra_predictions)
