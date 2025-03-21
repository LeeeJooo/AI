
from config import *

def train(model, X, y, epochs, batch_size, eval_interval, eval_X, eval_y, criterion, optimizer):
    train_loss, eval_loss = [], []
    train_acc, eval_acc = [], []

    n_samples = X.shape[0]

    for epoch in range(epochs):
        epoch_loss, correct_cnt = 0.0, 0

        for batch_idx_start in range(0, n_samples, batch_size):
            batch_idx_end = min(n_samples, batch_idx_start+batch_size)
            X_batch = X[batch_idx_start:batch_idx_end]  # (batch_size, n_channels, height, width) : MNIST (BATCH_SIZE, 1, 28, 28)
            y_batch = y[batch_idx_start:batch_idx_end]
            
            y_pred = model(X_batch)
            y_batch_one_hot = torch.zeros(y_pred.shape).to(device)
            y_batch_one_hot[torch.arange(y_pred.shape[0]), y_batch] = 1
            loss = criterion(y_pred, y_batch_one_hot)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # 기울기 초기화

            y_pred_label = torch.argmax(y_pred, dim=1)
            correct_cnt += (y_pred_label == y_batch).sum().item()

        avg_loss = epoch_loss/n_samples # 에포크 별, 배치 평균 손실
        acc = correct_cnt / n_samples
        train_loss.append(avg_loss)
        train_acc.append(acc)

        
        if (epoch+1)%eval_interval == 0:
            eval_loss_, eval_acc_ = evaluate(model, eval_X, eval_y, criterion)
            eval_loss.append(eval_loss_)
            eval_acc.append(eval_acc_)
        
            print(f"[Epoch {epoch+1}/{epochs}]  Train Loss: {avg_loss:.4f}    |   Train Acc: {acc:.2f}    |   Eval Loss: {eval_loss_:.4f}    |   Eval Acc: {eval_acc_:.2f}")
        
    return train_loss, train_acc, eval_loss, eval_acc

def evaluate(model, eval_X, eval_y, criterion):
    model.eval()

    with torch.no_grad():
        # LOSS
        y_pred = model(eval_X)

        y_one_hot = torch.zeros(y_pred.shape).to(device)
        y_one_hot[torch.arange(y_pred.shape[0]), eval_y] = 1

        loss = criterion(y_pred, y_one_hot)

        # ACCURACY
        correct_cnt = 0
        y_pred_label = torch.argmax(y_pred, dim=1)
        correct_cnt += (y_pred_label == eval_y).sum().item()
        acc = correct_cnt / y_pred.shape[0]

    model.train()

    return loss.item(), acc