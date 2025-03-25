from config import *
from model import *
from data import *
from train import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = RNN(input_size=n_letters, hidden_size=32, output_size=n_categories, num_layers=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss, train_acc = [], 0.0
    n_samples = 0
    for epoch in range(EPOCHS):
        for y, lines in category_lines.items():
            print(y)
            n_samples += len(lines)
            for line in lines:
                x_tensor = lineToTensor(line).to(device)
                y_idx = category_to_idx[y]
                
                loss, isCorrect = train(model, x_tensor, y_idx, criterion, optimizer)
                train_loss.append(loss)
                train_acc += isCorrect

        if (epoch+1)%1 == 0 :
            print(f"[Epoch {epoch+1}/{EPOCHS}]  Train Loss: {loss:.4f}    |   Train Acc: {train_acc/n_samples:.2f}")

