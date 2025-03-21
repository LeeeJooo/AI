import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualize(inputs, outputs1, outputs2):
    inputs=inputs.cpu().numpy().flatten()
    outputs1=outputs1.cpu().numpy().flatten()
    outputs2=outputs2.cpu().numpy().flatten()


    plt.figure(figsize=(8, 6))
    plt.title('Softmax vs Log Softmax')

    # Softmax 그래프 (파란색)
    plt.plot(inputs, outputs1, label='Softmax', color='blue', linewidth=2)
    
    # LogSoftmax 그래프 (빨간색)
    plt.plot(inputs, outputs2, label='LogSoftmax', color='red', linewidth=2)
    
    plt.xlabel('Inputs (x)')
    plt.ylabel('Output Value')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # 기준선
    plt.legend()
    plt.grid(True)
    plt.show()

MIN_VAL, MAX_VAL = -50.0, 50.0
DATASET_SIZE = 10000

softmax = nn.Softmax(dim=0).to(device)
log_softmax = nn.LogSoftmax(dim=0).to(device)

inputs = torch.linspace(MIN_VAL, MAX_VAL, DATASET_SIZE, dtype=torch.float32).view(DATASET_SIZE, 1).to(device)
# inputs = MIN_VAL + (MAX_VAL-MIN_VAL) * torch.rand(DATASET_SIZE, dtype=torch.float32).view(DATASET_SIZE, 1).to(device)
output1 = softmax(inputs)
output2 = log_softmax(inputs)

visualize(inputs, output1, output2)
