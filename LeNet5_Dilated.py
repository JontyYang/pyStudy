import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST


if __name__ == "__main__":
    # 使用fashionMiNist数据，准备训练集
    train_data = FashionMNIST(
        root="D:/SS/anacondaWork/pytorch/data/FashionMNIST",
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )

    # 定义一个数据加载器
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=False,  # 将shuffle设置为false，使得每个批次的训练中样本是固定的，这样有利于在训练中将数据切分为训练集和验证集
        num_workers=2
    )
    print("train_loader的batch数为:", len(train_loader))

    # 获取一个批次的数据进行可视化
    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break
        # 可视化
        batch_x = b_x.squeeze().numpy()  # 将维度为1的维度去除（图像的channel），即将（b，c, h, w）变为（b,h,w）
        batch_y = b_y.numpy()  # batch_y 是数字数组
        print(batch_y)
        # print(batch_x)
        class_label = train_data.classes
        print(class_label)
        plt.figure(figsize=(30, 10))
        for ii in range(len(batch_y)):
            plt.subplot(4, 16, ii + 1)
            plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
            plt.title(class_label[batch_y[ii]], size=15)
            plt.axis("off")
    #         plt.subplots_adjust(wspace = 0.05)

    # 对测试集进行处理
    test_data = FashionMNIST(
        root="D:/SS/anacondaWork/pytorch/data/FashionMNIST",
        train=False,
        download=False
    )

    test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
    print(test_data_x.shape)
    test_data_x = torch.unsqueeze(test_data_x, dim=1)
    print(test_data_x.shape)
    test_data_y = test_data.targets
    test_data_y


    # 搭建一个卷积神经网络
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            #         定义一个卷积层
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=2
                ),  # 1*28*28 -》 16*26*26
                nn.ReLU(),
                nn.AvgPool2d(
                    kernel_size=2,
                    stride=2
                )
            )
            #     定义第二个卷积层
            self.conv2 = nn.Sequential(
                nn.Conv2d(  # 16*13*13 -》 32*9*9
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    dilation=2
                ),
                nn.ReLU(),
                nn.MaxPool2d(  # 32*9*9 -》 32*4*4
                    kernel_size=2,
                    stride=2
                )
            )
            #     定义全连接层
            self.fc = nn.Sequential(
                nn.Linear(
                    in_features=32 * 4 * 4 ,
                    out_features=256
                ),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

        #     定义网络的向前传播路径
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            # print(x.shape)
            #         print(x.size())
            #         print(x.size(0))
            x = x.view(x.size(0), -1)  # 展平多维的卷积图层,第一个参数为保留的维度【64,32,6,6】-》【64,32*6*6】
            x
            # print(x.shape)
            output = self.fc(x)
            return output


    MyLeNet_Dilated = ConvNet()
    print(MyLeNet_Dilated)


    # x = torch.randn(1,1,28,28, requires_grad=True)
    # ConvNet(x)

    # 定义网络的训练过程函数
    def train_model(model,traindataloader,train_rate,criterion,optimizer,num_epoch=2):
        # model:网络模型；traindataloader:训练数据集；train_rate:训练集中训练数据与验证数据百分比；
    #     criterion:损失函数；optimizer:优化方法；num_epchs:训练轮数
        #计算训练使用的批次数
        batch_num = len(traindataloader)
        train_batch_num = round(batch_num * train_rate)
    #     复制模型的参数
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        train_loss_all = []
        train_acc_all = []
        val_loss_all = []
        val_acc_all = []
        since = time.time()
        for epoch in range(num_epoch):
            print('Epoch {}/{}'.format(epoch, num_epoch-1))
            print('-'*10)
            #每个epoch有两个训练阶段
            train_loss = 0.0
            train_corrects = 0
            train_num = 0
            val_loss = 0.0
            val_corrects = 0
            val_num = 0
            for step, (b_x, b_y) in enumerate(traindataloader):
    #             print("b-y:", b_y.shape)
                if step<train_batch_num:
                    model.train()   # 设置模型为训练模式
                    output = model(b_x)
                    pre_lab = torch.argmax(output, 1)
    #                 print("pre_lab", pre_lab.shape)
    #    损失函数计算损失
                    loss = criterion(output, b_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * b_x.size(0)
    #                 计算准确率,第一种是按照正确个数计算的，用的是准确率计算的
    #                 train_corrects += torch.sum(pre_lab == b_y.data)
                    train_corrects += accuracy_score(pre_lab, b_y)*b_x.size(0)
                    train_num += b_x.size(0)
                else:
                    model.eval() # 设置模型为验证模式
                    output = model(b_x)
                    pre_lab = torch.argmax(output, 1)
                    loss = criterion(output, b_y)
                    val_loss = loss.item()*b_x.size(0)
    #                 val_corrects += torch.sum(pre_lab == b_y.data)
                    val_corrects += accuracy_score(pre_lab, b_y)*b_x.size(0)
                    val_num += b_x.size(0)
            # 计算一个epoch中在训练集和验证集上的损失和精度
            train_loss_all.append(train_loss / train_num)
    #         通过前面的正确率计算第一种方法计算
    #         val_acc_all.append(val_corrects.double().item() / val_num)
    #       正确率计算第二种方法计算
            train_acc_all.append(train_corrects / train_num)
            val_loss_all.append(val_loss / val_num)
    #         val_acc_all.append(val_corrects.double().item() / val_num)
            val_acc_all.append(val_corrects / val_num)
            print('{} Train Loss: {:.4f}  Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
            print('{} Val Loss: {:.4f}  Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))
            # 拷贝模型最高精度下的参数(或者通过保存模型)
            if val_acc_all[-1] > best_acc:
                best_acc = val_acc_all[-1]
                best_model_wts = copy.deepcopy(model.state_dict())
            time_use = time.time() - since
            since = time.time()
            print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
            # 使用最好的模型参数
        model.load_state_dict(best_model_wts)
        # 将每个epoch的训练损失和精度放到数据表中输出
        train_process = pd.DataFrame(
            data = {"epoch": range(num_epoch),
                     "train_loss_all": train_loss_all,
                     "val_loss_all": val_loss_all,
                     "train_acc_all": train_acc_all,
                     "val_acc_all": val_acc_all
                     }
         )
        return model, train_process


    # 对模型进行训练
    optimizer = torch.optim.Adam(MyLeNet_Dilated.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss()
    # MyLeNet_Dilated_params = torch.load("D:/SS/anacondaWork/pytorch/model/LeNet5_Dilated_Drop_params.pkl")
    # MyLeNet_Dilated.load_state_dict(MyLeNet_Dilated_params)
    MyLeNet5_Dilated, train_process = train_model(MyLeNet_Dilated, train_loader, 0.8, criterion, optimizer,
                                                  num_epoch=2)


    def draw_history(train_process):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_process.epoch, train_process.train_loss_all, "ro-", label="Train Loss")
        plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="Val Loss")
        # plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(train_process.epoch, train_process.train_acc_all, "ro-", label="Train Acc")
        plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label="Val Acc")
        # plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("Acc")
        plt.show()


    draw_history(train_process)

    # 可视化模型的训练过程
    # 不知怎么的出现pickle错误，所以使用保存参数
    # torch.save(myLenet5, "../model/lenet5")
    torch.save(MyLeNet5_Dilated.state_dict(), "D:/SS/anacondaWork/pytorch/model/LeNet5_Drop_params.pkl")

    # 使用测试集
    MyLeNet5_Dilated.eval()
    output = MyLeNet5_Dilated(test_data_x)
    pre_lab = torch.argmax(output, 1)
    acc = accuracy_score(pre_lab, test_data_y)
    print("在测试集上的准确率为：", acc)

    # 计算混淆矩阵并可视化
    conf_mat = confusion_matrix(test_data_y, pre_lab)
    df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                rotation=0, ha="right")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()