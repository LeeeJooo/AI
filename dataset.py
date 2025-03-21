from config import *

data_path = './data'
download = not os.path.exists(os.path.join(data_path, 'MNIST'))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(data_path,
                            train=True,
                            download=download,
                            transform=transform)

test_data = datasets.MNIST(data_path,
                            train=False,
                            download=download,
                            transform=transform)

n_samples_train_data_total = len(train_data)
n_samples_train = int(n_samples_train_data_total * TRAIN_RATIO)
n_samples_eval = n_samples_train_data_total - n_samples_train
n_samples_test = len(test_data)

train_X = torch.stack([train_data[i][0] for i in range(0, n_samples_train)])
train_y = torch.tensor([train_data[i][1] for i in range(0, n_samples_train)])

eval_X = torch.stack([train_data[i][0] for i in range(n_samples_train, n_samples_train_data_total)])
eval_y = torch.tensor([train_data[i][1] for i in range(n_samples_train, n_samples_train_data_total)])

test_X = torch.stack([test_data[i][0] for i in range(n_samples_test)])
test_y = torch.tensor([test_data[i][1] for i in range(n_samples_test)])