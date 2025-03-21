from config import *

def visualize_results(epochs, train_loss, train_acc, eval_loss, eval_acc):
    _, axes = plt.subplots(1, 2, figsize=(20,8))
    axes[0].set_title('TRAIN RESULT')
    ax_train_loss = axes[0]
    ax_train_loss.plot(range(epochs), train_loss, label="LOSS", color='purple', linewidth=3)
    ax_train_loss.set(xlabel='EPOCH', ylabel='LOSS')
    ax_train_loss.grid()
    ax_train_loss.legend(loc="upper left")

    ax_train_acc = ax_train_loss.twinx()
    ax_train_acc.plot(range(epochs), train_acc, label="ACC", color='green', linewidth=3)
    ax_train_acc.set(xlabel='EPOCH', ylabel='ACC')
    ax_train_acc.grid()
    ax_train_acc.legend(loc="upper right")

    axes[1].set_title('VALIDATE RESULT')
    ax_eval_loss = axes[1]
    ax_eval_loss.plot(range(len(eval_loss)), eval_loss, label="LOSS", color='purple', linewidth=3)
    ax_eval_loss.set(xlabel='EPOCH', ylabel='LOSS')
    ax_eval_loss.grid()
    ax_eval_loss.legend(loc="upper left")

    ax_eval_acc = ax_eval_loss.twinx()
    ax_eval_acc.plot(range(len(eval_acc)), eval_acc, label="ACC", color='green', linewidth=3)
    ax_eval_acc.set(xlabel='EPOCH', ylabel='ACC')
    ax_eval_acc.grid()
    ax_eval_acc.legend(loc="upper right")

    plt.show()

def visualize_grad_cam():
    pass