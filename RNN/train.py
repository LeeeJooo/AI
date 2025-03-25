# 학습 실행
from config import *

def train(model, X, y_label, criterion, optimizer):
    y_pred = model(X)
    y_pred_label = torch.argmax(y_pred)
    isCorrect=0
    if y_pred_label==y_label : isCorrect = 1

    y_tensor = torch.zeros(y_pred.shape).to(device)
    y_tensor[y_label] = 1
    loss = criterion(y_pred, y_tensor)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), isCorrect

    # for epoch in range(epochs):
        # epoch_loss = 0.0
        # correct_cnt = 0

        # for batch_start_idx in range(0, n_samples):
        #     batch_end_idx = min(n_samples, batch_start_idx+batch_size)
        #     X_batch = X[batch_start_idx:batch_end_idx]
        #     y_batch = y[batch_start_idx:batch_end_idx]
        #     y_pred = model(X_batch, y_batch)

        #     loss = criterion(y_pred, y_batch)
        #     epoch_loss += loss.item()
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()

        #     y_pred_label = torch.argmax(y_pred, dim=1)
        #     y_label = torch.argmax(y_batch, dim=1)
        #     correct_cnt = (y_pred_label == y_label).sum().item()
        
        # avg_loss = epoch_loss/n_samples # 에포크 별, 배치 평균 손실
        # acc = correct_cnt / n_samples

