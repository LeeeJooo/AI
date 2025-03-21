from CNN import *
from config import *
from pipeline import *
from visualize import *
from dataset import train_X, train_y, eval_X, eval_y, test_X, test_y

if __name__ == "__main__":
    print('RUN CNN')

    # SETTING
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_X, train_y, eval_X, eval_y, test_X, test_y = train_X.to(device), train_y.to(device), eval_X.to(device), eval_y.to(device), test_X.to(device), test_y.to(device)

    # TRAIN
    print('START TO TRAIN CNN MODEL')
    train_loss, train_acc, eval_loss, eval_acc = train(model, train_X, train_y, EPOCHS, BATCH_SIZE, EVAL_INTERNAL, eval_X, eval_y, criterion, optimizer)
    print('FINISH TRAINING CNN')

    # SAVE
    os.makedirs(save_dir, exist_ok=True)
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    filename = os.path.join(save_dir, f'CNN_MNIST_{EPOCHS}epochs_{current_time}.pth')
    torch.save(model.state_dict(), filename)
    # torch.save(model, filename)
    print(f'Model saved as: {os.path.abspath(filename)}')
    
    # TEST
    test_loss, test_acc = evaluate(model, test_X, test_y, criterion)
    print(f'TEST LOSS = {test_loss:.2f}, TEST ACCURACY = {test_acc:.2f}')
    
    # VISUALIZE
    visualize_results(EPOCHS, train_loss, train_acc, eval_loss, eval_acc)