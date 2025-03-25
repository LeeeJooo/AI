# RNN 정의
from config import *

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.Wx_1 = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.Wx_2 = nn.Parameter(torch.randn(num_layers, self.hidden_size, self.hidden_size))
        self.Wh = nn.Parameter(torch.randn(num_layers, self.hidden_size, self.hidden_size))
        self.b = nn.Parameter(torch.randn(num_layers, self.hidden_size))
        self.Wo_1 = nn.Parameter(torch.randn(self.hidden_size, 1))
        self.Wo_2 = nn.Parameter(torch.randn(1, 1))
        self.Wo = nn.Parameter(torch.randn(self.hidden_size, self.output_size))
    
        self.tanh = nn.Tanh()
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        seq_len, batch_size, _ = X.shape
        h_t = torch.zeros(self.num_layers, seq_len, batch_size, self.hidden_size).to(device)
        h_t = h_t.contiguous()
        
        # HIDDEN LAYER
        for seq_i in range(seq_len):
            is_first_layer = True
            weighted_h_t_minus_1 = torch.zeros(batch_size, self.hidden_size)
            for layer_i in range(self.num_layers):
                '''
                [ 1st Layer ]
                x[seq_i]            @ Wx[layer_i] : (batch_size, input_size)  @ (input_size, hidden_size)  -> (batch_size, hidden_size)
                h[layer_i][seq_i-1] @ Wh[layer_i] : (batch_size, hidden_size) @ (hidden_size, hidden_size) -> (batch_size, hidden_size)
                
                [ 2nd Layer ~ ]
                h[layer_i-1][seq_i] @ Wx[layer_i]  : (batch_size, input_size)  @ (input_size, hidden_size)  -> (batch_size, hidden_size)
                h[layher_i][seq_i-1] @ Wh[layer_i] : (batch_size, hidden_size) @ (hidden_size, hidden_size) -> (batch_size, hidden_size)
                '''
                if is_first_layer:
                    weighted_x = X[seq_i]@self.Wx_1                         # (batch_size,input_size) @ (input_size,hidden_size)
                    is_first_layer = False
                else:
                    weighted_x = h_t[layer_i-1][seq_i]@self.Wx_2[layer_i]   # (batch_size,hidden_size) @ (hidden_size, hidden_size)
                
                if seq_i == 0:
                    weighted_h_t_minus_1 = torch.zeros(batch_size, self.hidden_size).to(device)
                else:
                    weighted_h_t_minus_1 = h_t[layer_i][seq_i-1]@self.Wh[layer_i]

                h_t[layer_i] = self.tanh(weighted_x + weighted_h_t_minus_1 + self.b[layer_i]).clone()
        
        # OUTPUT LAYER
        output = h_t[-1][-1]            # (batch_size, self.hidden_size)
        output = output @ self.Wo       # (batch_size, self.hidden_size) @ (hidden_size, output_size) -> (batch_size, output_size)
        output = self.logSoftmax(output)
        output = output.squeeze()      # (output_size)

        return output