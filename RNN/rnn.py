import torch
import torch.nn as nn

class RNNCell_Encoder(nn.Module):
    def __init__(self, config_obj):
        super(RNNCell_Encoder, self).__init__()
        self.device = config_obj.device
        self.rnn = nn.RNNCell(input_size=config_obj.rnn_config.input_dim,       # 입력 x_t의 feature 크기 : 단어 임베딩 차원
                              hidden_size=config_obj.rnn_config.hidden_size,    # 은닉 상태 h_t의 크기 : RNN이 기억할 정보의 차원
                              bias=config_obj.rnn_config.bias,                  # 편향 사용 여부
                              nonlinearity=config_obj.rnn_config.nonlinearity,  # RNN 활성화 함수
                              device=self.device,                               # 초기화할 디바이스 (cpu or cuda)
                              dtype=torch.float32                               # weight 초기화 dtype
                              )
        
    def forward(self, inputs, hidden_size): # config:hidden_size, device
        batch_size = inputs.shape[0]
        hidden_state = torch.zeros((batch_size, hidden_size)).to(self.device)

        for word in inputs:
            hidden_state = self.rnn(input=word, hidden=hidden_state)
        
        return hidden_state

class RNN(nn.Module):
    def __init__(self, config_obj):
        super(RNN, self).__init__()
        self.config_obj = config_obj
        self.rnn_config_obj = config_obj.rnn_config
        
        self.em = nn.Embedding(num_embeddings=self.config_obj.rnn_config.num_embeddings,
                               embedding_dim=self.config_obj.rnn_config.embedding_dim)
        self.rnn = RNNCell_Encoder(config_obj)
        self.fc1 = nn.Linear(self.rnn_config_obj.hidden_size, 256)
        self.fc2 = nn.Linear(256, 3)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.em(x)
        x = self.rnn(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

if __name__=="__main__":
    from config import Config
    config_obj = Config()
    model = RNN(config_obj=config_obj).to(device=config_obj.device)