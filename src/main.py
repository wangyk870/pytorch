import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from  src.ExampleNet import ExampleNet

def main():
    print("Is GPU available : {}".format(torch.cuda.is_available()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 20
    batch_size = 500

    net = ExampleNet().to(device)
    loss_fun = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())

    transforms_train = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = datasets.CIFAR10("datasets/",
                     train=True,
                     download=True,
                     transform=transforms_train)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    testdataset = datasets.CIFAR10("datasets/",
                                   train=False,
                                   download=False,
                                   transform=transform_test)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=False)

    for i in range(epoch):
        net.train()
        print("epoch:{}".format(i))
        for j,(input, target) in enumerate(dataloader):
            input = input.to(device)
            output = net(input)
            target = torch.zeros(target.size(0), 10).scatter(1, target.view(-1, 1), 1).to(device)
            loss = loss_fun(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % 10 == 0:
                print("[epoch - {0} - {1}/{2}] loss:{3}".format(i, j, len(dataloader), loss.float()))
                # print("loss:{}".format(loss.float()))
        with torch.no_grad():
            net.eval()
            correct = 0.
            total = 0.
            for input, target in testdataloader:
                input, target = input.to(device), target.to(device)
                output = net(input)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()
                accuracy = correct.float() / total
            print("[epoch - {0}Accuracy:{1}%]".format(i+1, (100 * accuracy)))

    torch.save(net, "models/net.pth")
