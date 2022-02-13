import torch
import torchvision.transforms as transforms
from PIL import Image

def core(net, img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    img = transform(img).unsqueeze(0)
    output = net(img.to(device))
    index = torch.argmax(output).item()
    list = ['飞机', '小汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    print(list[index])

if __name__ == '__main__':
    net = torch.load("models/net.pth")
    core(net, "test_data/pic01.jpg")
