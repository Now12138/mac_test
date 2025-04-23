# 引入依赖库
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn import preprocessing
from models import *
from utils import *
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

# 必要参数定义
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 训练设备,如果NVIDIA GPU已配置，会自动使用GPU训练
print(device)
train_ratio = 0.7  # 训练集比例
val_ratio = 0.1  # 验证集比例
test_ratio = 0.2  # 测试集比例
batch_size = 50  # 批大小，若用CPU，建议为1
input_length = 10  # 每个batch的输入数据长度
output_length = 1  # 每个batch的输出数据长度，这里只能单步预测
loss_function = 'MSE'  # 损失函数定义
learning_rate = 0.01  # 基础学习率
weight_decay = 0.001  # 权重衰减系数
num_blocks = 2 # lstm堆叠次数
dim = 64  # 隐层维度
scalar = True  # 是否使用归一化
scalar_contain_labels = True  # 归一化过程是否包含目标值的历史数据
interval_length = 50000  # 预测数据长度，最长不可以超过总数据条数
target_value = 'Tlog (degC)'  # 需要预测的列名
# 多步，单步标签 
if output_length > 1:
    forecasting_model = 'multi_steps'
else:
    forecasting_model = 'one_steps'

#  读取数据
df = pd.read_excel("mpi_roof_2024.xlsx")
df = df[:interval_length]

#df['year'] = df['date'].dt.year
#df['month'] = df['date'].dt.month
#df['day'] = df['date'].dt.day
#df['hour'] = df['date'].dt.hour
#df['minute'] = df['date'].dt.minute
#df=df.drop(columns='date')

features_num = df.shape[1]  #特征总数
print("******************************************")
labels_ = df[target_value].values
print(labels_.shape)
if features_num > 1:
    features_ = df.values
else:
    features_ = df[target_value].values

# 初步划分训练集、验证集、测试集
split_train_val, split_val_test = int(len(features_)*train_ratio),\
                                  int(len(features_)*train_ratio)+int(len(features_)*val_ratio)

#  数据标准化
if scalar:
    #min-max scalar
    train_features_ = features_[:split_train_val]#训练集的特征
    val_test_features_ = features_[split_train_val:]#验证集＋测试集的特征
    scalar = preprocessing.MinMaxScaler()
    if features_num == 1:
        train_features_ = np.expand_dims(train_features_, axis=1)
        val_test_features_ = np.expand_dims(val_test_features_, axis=1)
    train_features_ = scalar.fit_transform(train_features_)
    val_test_features_ = scalar.transform(val_test_features_)
    # 重新将数据进行拼接
    features_ = np.vstack([train_features_, val_test_features_])
    if scalar_contain_labels:
        QQ = df.columns.get_loc(target_value)
        labels_ = features_[:, QQ]
        print(labels_[0:20])
    #features_=np.delete(features_,QQ,axis=1)

if len(features_.shape) == 1:
    features_ = np.expand_dims(features_,0).T
features, labels = get_rolling_window_multistep(output_length, 0, input_length,
                                                features_.T, np.expand_dims(labels_, 0))
#  构建数据集
labels = torch.squeeze(labels, dim=1)#若第1维的大小为1，则删除此维，否则不变
features = features.to(torch.float32)
labels = labels.to(torch.float32)
train_features, train_labels = features[:split_train_val], labels[:split_train_val]
val_features, val_labels = features[split_train_val:split_val_test], labels[split_train_val:split_val_test]
test_features, test_labels = features[split_val_test:], labels[split_val_test:]

#  数据管道构建，此处采用torch高阶API
train_Datasets = TensorDataset(train_features.to(device), train_labels.to(device))
train_Loader = DataLoader(batch_size=batch_size, dataset=train_Datasets)
val_Datasets = TensorDataset(val_features.to(device), val_labels.to(device))
val_Loader = DataLoader(batch_size=batch_size, dataset=val_Datasets)
test_Datasets = TensorDataset(test_features.to(device), test_labels.to(device))
test_Loader = DataLoader(batch_size=batch_size, dataset=test_Datasets)

#  模型定义
LSTMMain_model = LSTMMain(input_size=features_num, output_len=output_length,
                                  lstm_hidden=dim, lstm_layers=num_blocks, batch_size=batch_size, device=device)
modle_num=4
LSTMMain_model.to(device)
if loss_function == 'MSE':
    loss_func = nn.MSELoss(reduction='mean')
#01：.69，2：0.73，3：负娃，4：负娃
#  训练代数定义
epochs = 58
#  优化器定义，学习率衰减定义
optimizer = torch.optim.AdamW(LSTMMain_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//3, eta_min=0.00001)

#  训练及验证循环
print("——————————————————————Training Starts——————————————————————")
for epoch in range(epochs):
    # 训练
    LSTMMain_model.train()
    train_loss_sum = 0
    step = 1
    for step, (feature_, label_) in enumerate(train_Loader):
        optimizer.zero_grad()
        #print(feature_.shape)
        feature_ = feature_.permute(0,2,1)
        prediction = LSTMMain_model(feature_,modle_num)
        loss = loss_func(prediction, label_)
        loss.backward()
        torch.nn.utils.clip_grad_norm(LSTMMain_model.parameters(), 0.15)#梯度裁剪
        optimizer.step()
        train_loss_sum+=loss.item()
    print("epochs = " + str(epoch))
    print('train_loss = ' + str(train_loss_sum))

    #  验证
    LSTMMain_model.eval()
    val_loss_sum = 0
    val_step = 1
    for val_step, (feature_, label_) in enumerate(val_Loader):
        feature_ = feature_.permute(0, 2, 1)
        with torch.no_grad():
            prediction = LSTMMain_model(feature_,modle_num)
            val_loss = loss_func(prediction, label_)
        val_loss_sum += val_loss.item()
    if epoch == 0:
        val_best = val_loss_sum
        print('val_loss = ' + str(val_loss_sum))
        torch.save(LSTMMain_model.state_dict(), './weights/model_LSTMMain_weights')  # 保存最好权重
        print("val_best change")
    else:
        print('val_loss = ' + str(val_loss_sum))
        if val_best > val_loss_sum:
            val_best = val_loss_sum
            torch.save(LSTMMain_model.state_dict(), './weights/model_LSTMMain_weights')  # 保存最好权重
            print("val_best change")
print("best val loss = " + str(val_best))
print("——————————————————————Training Ends——————————————————————")

#  测试集预测
LSTMMain_model.load_state_dict(torch.load('./weights/model_LSTMMain_weights'))  # 调用权重
test_loss_sum = 0
#  测试集inference
print("——————————————————————Testing Starts——————————————————————")
for step, (feature_, label_) in enumerate(test_Loader):
    feature_ = feature_.permute(0, 2, 1)
    with torch.no_grad():
         if step ==0:
            prediction = LSTMMain_model(feature_,modle_num)
            pre_array = prediction.cpu()
            label_array = label_.cpu()
            loss = loss_func(prediction, label_)
            test_loss_sum += loss.item()
         else:
            prediction = LSTMMain_model(feature_,modle_num)
            pre_array = np.vstack((pre_array, prediction.cpu()))
            label_array = np.vstack((label_array, label_.cpu()))
            loss = loss_func(prediction, label_)
            test_loss_sum += loss.item()
print("test loss = " + str(test_loss_sum))
print("——————————————————————Testing Ends——————————————————————")

# 数据后处理，单步预测绘制全部预测值的图像，多步预测仅绘制第一个batch的输出图像
#  逆归一化过程及绘制图像
print("——————————————————————Post-Processing——————————————————————")
if scalar_contain_labels and scalar:
    pre_inverse = []
    test_inverse = []
    if features_num == 1 and output_length == 1:
        for pre_slice in range(pre_array.shape[0]):
            pre_inverse_slice = scalar.inverse_transform(np.expand_dims(pre_array[pre_slice,:], axis=1))
            test_inverse_slice = scalar.inverse_transform(np.expand_dims(label_array[pre_slice,:], axis=1))
            pre_inverse.append(pre_inverse_slice)
            test_inverse.append(test_inverse_slice)
        pre_array = np.array(pre_inverse).squeeze(axis=-1)
        test_labels = np.array(test_inverse).squeeze(axis=-1)
    elif features_num > 1:
        if isinstance(pre_array, np.ndarray):
            pre_array = torch.from_numpy(pre_array)

        for pre_slice in range(pre_array.shape[0]):
            pre_num = pre_array[0].shape[0]
            test_num = test_labels[0].shape[0]
            pre_inverse_slice = scalar.inverse_transform(torch.cat((torch.zeros(pre_num, QQ),
                                                                    torch.unsqueeze(pre_array[pre_slice], dim=1),
                                                                    torch.zeros(pre_num, features_num - QQ-1)), 1))[:,QQ]
            test_inverse_slice = scalar.inverse_transform(torch.cat((torch.zeros(test_num, QQ),
                                                                     torch.unsqueeze(test_labels[pre_slice], dim=1),
                                                                     torch.zeros(test_num, features_num - QQ-1)), 1))[:, QQ]
            pre_inverse.append(pre_inverse_slice)
            test_inverse.append(test_inverse_slice)

        pre_array = np.array(pre_inverse)
        test_labels = np.array(test_inverse)
        # print(pre_array)

    else:
        for pre_slice in range(pre_array.shape[0]):
            pre_inverse_slice = scalar.inverse_transform(np.expand_dims(pre_array[pre_slice,:], axis=1))
            test_inverse_slice = scalar.inverse_transform(np.expand_dims(label_array[pre_slice,:], axis=1))
            pre_inverse.append(pre_inverse_slice)
            test_inverse.append(test_inverse_slice)
        pre_array = np.array(pre_inverse).squeeze(axis=-1)
        test_labels = np.array(test_inverse).squeeze(axis=-1)
    plt.figure(figsize=(40,20))
    if forecasting_model == 'multi_steps':
        plt.plot(pre_array[0], 'g')
        plt.plot(test_labels[0], "r")
        plt.legend(["forecast", "actual"], loc='upper right')
        plt.show()
    else:
        plt.plot(pre_array, 'g')
        plt.plot(test_labels, "r")
        plt.legend(["forecast", "actual"], loc='upper right')
        plt.show()
    #  计算衡量指标
    MSE_l = mean_squared_error(test_labels, pre_array)
    MAE_l = mean_absolute_error(test_labels, pre_array)
    MAPE_l = mean_absolute_percentage_error(test_labels, pre_array)
    R2 = r2_score(test_labels, pre_array)
    print('MSE loss=%s'%MSE_l)
    print('MAE loss=%s'%MAE_l)
    print('MAPE loss=%s'%MAPE_l)
    print('R2=%s'%R2)

else:
    plt.figure(figsize=(40,20))
    if forecasting_model == 'multi_steps':
        plt.plot(pre_array[0], 'g')
        plt.plot(test_labels[0].cpu(), "r")
        plt.legend(["forecast", "actual"], loc='upper right')
        plt.show()
    else:
        plt.plot(pre_array, 'g')
        plt.plot(test_labels.cpu(), "r")
        plt.legend(["forecast", "actual"], loc='upper right')
        plt.show(block = True)
    MSE_l = mean_squared_error(test_labels.cpu(), pre_array)
    MAE_l = mean_absolute_error(test_labels.cpu(), pre_array)
    MAPE_l = mean_absolute_percentage_error(test_labels.cpu(), pre_array)
    R2 = r2_score(test_labels.cpu(), pre_array)
    print('MSE loss=%s'%MSE_l)
    print('MAE loss=%s'%MAE_l)
    print('MAPE loss=%s'%MAPE_l)
    print('R2=%s'%R2)

file_path = "./weights/model_LSTMMain_weights"
# 直接覆盖文件内容
with open(file_path, "wb") as file:
    pass
