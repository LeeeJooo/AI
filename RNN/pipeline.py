import torch

class Pipeline:
    def __init__(self, config_obj, dataset, preprocess_fn, model, optimizer, loss_function):
        self.config_obj = config_obj
        self.dataset = dataset
        num_embeddings, self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = preprocess_fn(dataset)
        self.config_obj.rnn_config.set_num_embeddings(num_embeddings)
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train(self):
        epochs = self.config_obj.pipeline_config.epochs
        batch_size = self.config_obj.pipeline_config.batch_size
        train_loss = []
        train_acc = []
        valid_loss = []
        valid_acc = []

        for epoch in range(epochs):
            train_x, train_y, valid_x, valid_y = None, None, None, None # batch로 나누기
            epoch_loss, epoch_acc, epoch_valid_acc, epoch_valid_loss = self._train(epoch, train_x, train_y, valid_x, valid_y)
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
            valid_loss.append(epoch_valid_loss)
            valid_acc.append(epoch_valid_acc)

    def _train(self, epoch, train_x, train_y, valid_x, valid_y):
        correct = 0
        total = 0
        running_loss = 0

        for x, y in zip(train_x, train_y):
            x, y = x.to(self.config_obj.device), y.to(self.config_obj.device)
            y_pred = self.model(x)
            loss = self.loss_function(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                y_pred = torch.argmax(y_pred, dim=1)
                correct += (y_pred==y).sum().item()
                total += y.size(0)
                running_loss += loss.item()

        epoch_loss = running_loss/len(train_x)
        epoch_acc = correct/total

        valid_correct = 0
        valid_total = 0
        valid_running_loss = 0

        self.model.eval()
        with torch.no_grad():
            for x, y in zip(valid_x, valid_y):
                x, y = x.to(self.config_obj.device), y.to(self.config_obj.device)
                y_pred = self.model(x)
                loss = self.loss_function(y_pred, y)
                y_pred = torch.argmax(y_pred, dim=1)
                valid_correct += (y_pred==y).sum().item()
                valid_total += y.size(0)
                valid_running_loss += loss.item()
        
        epoch_valid_loss = valid_running_loss/len(valid_x)
        epoch_valid_acc = valid_correct/valid_total

        print(f'epoch : {epoch:.5d}, loss : {loss:.3f}, accuracy : {epoch_acc:.3f}, valid_loss : {epoch_valid_loss:.3f}, valid_accuracy : {epoch_valid_acc:.3f}')

        return epoch_loss, epoch_acc, epoch_valid_acc, epoch_valid_loss
    
    def evaluate(self):
        pass
    
if __name__ == "__main__":
    from config import Config
    from dataset import Dataset
    from rnn import RNN
    import torch.optim as optim
    import torch.nn as nn

    config_obj = Config()
    dataset_obj = Dataset()
    vocab_size, train_x, train_y, valid_x, valid_y, test_x, test_y = dataset_obj.preprocess_imdb(config_obj, dataset_obj.dataset)
    config_obj.rnn_config.set_num_embeddings(vocab_size)
    model = RNN(config_obj)
    optimizer = optim.Adam(model.parameters(), lr=config_obj.pipeline_config.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    pipeline = Pipeline(config_obj, dataset_obj.dataset, dataset_obj.preprocess_imdb, model, optimizer, loss_function)

    pipeline.train()
    pipeline.evaluate()