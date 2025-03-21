from CNN import *
from config import *
from dataset import train_data
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

model = CNN().to(device)
model.load_state_dict(torch.load('models/CNN_MNIST_50epochs_20250321_1459.pth'))
model.eval()
dataset = train_data
print('START GRAD-CAM WITH CNN MNIST MODEL')

target_layers = [model.conv2]

# 시각화 설정
h, w = 5, 5
fig, axes = plt.subplots(h, w*2, figsize=(20, 10))
axes = axes.flatten()

for i in range(0, h*2*w, 2):
    img, label = dataset[i]
    
    # Grad-CAM 입력은 [batch_size, C, H, W] 형태 필요
    input_tensor = img.unsqueeze(0).float()
    
    # Grad-CAM 타겟 설정
    targets = [ClassifierOutputTarget(label)]
    
    # Grad-CAM 실행
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets) # (1, 28, 28)
    
    # 히트맵 생성 및 시각화
    img = (img - img.min()) / (img.max() - img.min())  # [0, 1]로 정규화
    visualization = show_cam_on_image(
        img.permute(1, 2, 0).numpy(),  # (1, 28, 28) → (28, 28, 1)
        grayscale_cam[0, :],
        use_rgb=True
    )
    
    # 이미지 표시
    axes[i].imshow(grayscale_cam[0,:])
    axes[i].axis("off")

    axes[i+1].imshow(visualization) # (28, 28, 3)
    axes[i+1].axis("off")

plt.show()