import torch

class Config:
    def __init__(self):
        self.database_config = DatasetConfig()
        self.rnn_config = RNNConfig()
        self.pipeline_config = PipelineConfig()
        self.device = self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DatasetConfig:
    def __init__(self):
        self.imdb_train_valid_ratio = 0.8   # 8:2
        self.imdb_tokenizer = "bert-base-uncased"

class RNNConfig:
    def __init__(self):
        # RNNCell
        self.input_dim = 0
        self.hidden_size = 0
        self.bias = True
        self.nonlinearity = 'tanh'  # tanh(default), relu

        # Embedding
        self.num_embeddings = None  # vocab 크기. 전체 단어 수.
        self.embedding_dim = 0      # 각 단어를 표현할 벡터 크기.

        # Linear (1)

        # Linear (2)

    def set_num_embeddings(self, num_embeddings):
        self.num_embeddings = num_embeddings

class PipelineConfig:
    def __init__(self):
        self.epochs = 1
        self.batch_size = 100
        self.learning_rate = 0.0001