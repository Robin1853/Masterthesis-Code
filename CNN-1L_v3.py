import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import operator

from math import pi
import sys
import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import torch.utils.data as data_utils
from torchinfo import summary

from torch.utils.data import DataLoader, DistributedSampler

import defs




def norm01(tensor):
    #normed_tensor = tensor / 255 * (2 * pi)
    normed_tensor = tensor / 255
    return normed_tensor




def norm02(tensor):
    #normed_tensor = tensor * (2 * pi)
    normed_tensor = tensor
    return normed_tensor


def data_loading(datasource, batchsize, trainsize):
    if datasource == "MNIST":

        trainset = torchvision.datasets.MNIST(root='/ifxhome/faierr/projects/qml-cybrex/models/data/', train=True, download=True,
                                              transform=transforms.Compose([torchvision.transforms.ToTensor(), norm02]))
        testset = torchvision.datasets.MNIST(root='/ifxhome/faierr/projects/qml-cybrex/models/data/', train=False, download=True,
                                             transform=transforms.Compose([torchvision.transforms.ToTensor(), norm02]))

        tr_12k = torch.utils.data.Subset(trainset, range(trainsize))
        te_2k = data_utils.Subset(testset, range(int(trainsize / 6)))

        trainloader = DataLoader(dataset=tr_12k, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(dataset=te_2k, batch_size=batchsize, shuffle=False)

    elif datasource == "breastmnist":

        trainset = np.load('/ifxhome/faierr/projects/qml-cybrex/models/data/.medmnist/breastmnist/train_images.npy')
        # # trainset_at_eval = np.load('C:\Users\FaierR\models\data\breastmnist/val_images.npy')
        testset = np.load('/ifxhome/faierr/projects/qml-cybrex/models/data/.medmnist/breastmnist/test_images.npy')
        #
        trainlab = np.load('/ifxhome/faierr/projects/qml-cybrex/models/data/.medmnist/breastmnist/train_labels.npy')
        # # trainlab_at_eval = np.load('C:\Users\FaierR\models\data\breastmnist/val_labels.npy')
        testlab = np.load('/ifxhome/faierr/projects/qml-cybrex/models/data/.medmnist/breastmnist/test_labels.npy')

        # trainset = np.load('C:/Users/FaierR/models/data/breastmnist/train_images.npy')
        # trainset_at_eval = np.load('C:\Users\FaierR\models\data\breastmnist/val_images.npy')
        # testset = np.load('C:/Users/FaierR/models/data/breastmnist/test_images.npy')

        # trainlab = np.load('C:/Users/FaierR/models/data/breastmnist/train_labels.npy')
        # trainlab_at_eval = np.load('C:\Users\FaierR\models\data\breastmnist/val_labels.npy')
        # testlab = np.load('C:/Users/FaierR/models/data/breastmnist/test_labels.npy')

        # trainset = np.array([np.array([trainset[i],trainlab[i]],dtype=object) for i in range(len(trainset))])
        # trainset_at_eval = [np.array([trainset_at_eval[i],
        # trainlab_at_eval[i]],dtype=object) for i in range(len(trainset))]
        # testset = np.array([np.array([testset[i],testlab[i]],dtype=object) for i in range(len(testset))])

        trainset = torch.Tensor(norm01(trainset))
        trainlab = torch.Tensor(trainlab)
        testset = torch.Tensor(norm01(testset))
        testlab = torch.Tensor(testlab)

        trainset = torch.utils.data.TensorDataset(trainset, trainlab)
        testset = torch.utils.data.TensorDataset(testset, testlab)

        tr_12k = torch.utils.data.Subset(trainset, range(trainsize))
        te_2k = data_utils.Subset(testset, range(int(trainsize / 6)))

        trainloader = DataLoader(dataset=trainset, batch_size=batch_size)
        testloader = DataLoader(dataset=testset, batch_size=batchsize, shuffle=False)

    else:
        sys.exit("Error: no source given")

    return trainloader, testloader, tr_12k, te_2k
    # return tr_12k, te_2k


class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(2, 2), stride=2, padding='valid')
        self.act1 = nn.ReLU()
        self.flat = nn.Flatten()
        self.clayer_1 = nn.Linear(14 * 14*4, 30)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(30, 10)

        #self.to(dev)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        def rest_forward(x):
            x = self.flat(x)
            # print(x.size())
            x = self.clayer_1(x)
            x = self.act3(x)
            x = self.drop3(x)
            x = self.fc4(x)
            return x

        x = torch.utils.checkpoint.checkpoint(rest_forward, x)

        return x


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)


def initialization(j):
    print("Seed:", j)
    model.apply(weights_init)


#     for parameter in model.parameters():
#         print(parameter.data, parameter.data.size())
#    summary(model, input_size=(batch_size, 1, 28, 28))


def train(net_model, data_loader):
    global loss
    optimizer = optim.Adam(net_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    step_loss = []
    count = 0
    t_acc = 0
    train_acc = []

    for data, labels in tqdm.tqdm(data_loader):
        data, labels = data, labels
        optimizer.zero_grad()
        labels = labels.long()
        pred = net_model(data)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        step_loss.append(loss.item())
        t_acc += (torch.argmax(pred, 1) == labels).float().sum()
        train_acc.append(t_acc)
        count += len(labels)
    t_acc /= count

    print(f'Accuracy: {t_acc * 100:.2f}%, Loss: {loss.item():.4f}')
    # print(np.array(step_loss[-1]), np.array(train_acc).mean())
    return np.array(step_loss).mean(), np.array(train_acc).mean()
    #return torch.mean(torch.Tensor(step_loss)).cpu().numpy(), torch.mean(torch.tensor(train_acc)).cpu().numpy()


def test(model, data_loader):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    global val_loss

    acc = 0
    count = 0
    # model.eval()
    val_step_loss = []
    val_acc = []

    with torch.no_grad():
        # global val_loss
        for data, labels in tqdm.tqdm(data_loader):
            #data, labels = data.to(dev), labels.to(dev)
            labels = labels.long()
            pred = model(data)
            val_loss = criterion(pred, labels)

            val_step_loss.append(val_loss.item())
            acc += (torch.argmax(pred, 1) == labels).float().sum()
            val_acc.append(acc)
            count += len(labels)
    acc /= count
    print(f'Val Accuracy: {acc * 100:.2f}%, Validation Loss: {val_loss.item():.4f}')
    return np.array(val_step_loss).mean(), np.array(val_acc).mean()
    #return torch.mean(torch.Tensor(val_step_loss)).cpu().numpy(), torch.mean(torch.tensor(val_acc)).cpu().numpy()


torch.manual_seed(0)

base_seed = torch.randint(0, 100, (20,), dtype=torch.int)

# print(base_seed[19])

# ##############################################
# ##input parameters to change##################
# ##############################################

batches = [8, 16, 32, 64]

train_size = [10, 100, 1000, 10000]
#train_size = [60000] #[10000, 60000]
data_source = "MNIST"

n_epochs = 20
n_seeds = 10

ent1 = "CNN-1L"
identity = "nopool"
# bs = f"bs{batch_size}"

directory: str = f"/ifxhome/faierr/projects/qml-cybrex/models/results/classical_results/{ent1}_{identity}_{data_source}/"
os.makedirs(os.path.dirname(directory), exist_ok=True)
#qml.draw_mpl(qnode2, decimals=2)(inputs=['inp1', 'inp2', 'inp3', 'inp4'], weights=[['w1', 'w2', 'w3', 'w4']])
# weights=[['w11', 'w12', 'w13', 'w14'],['w21', 'w22', 'w23', 'w24']])
# weights=[[['w111', 'w112', 'w113'],['w121', 'w122', 'w123'],['w131', 'w132', 'w133'],['w141', 'w142', 'w143']],[['w211', 'w212', 'w213'],['w221', 'w222', 'w223'],['w231', 'w232', 'w233'],['w241', 'w242', 'w243']]])
# weights=[['w1', 'w2', 'w3', 'w4']])

#plt.savefig(f"{directory}Circuit.png")
#plt.show()

defs.tic()
for size in train_size:
    print(size)
    for batch_size in batches:
        bs = f"bs{batch_size}"
        print(batch_size)

        #size = train_size

        parameter_in = []
        parameter_out = []
        plot_data_sum = []
        plot_data = []

        trainingEpoch_loss_sum = []
        validationEpoch_loss_sum = []
        train_accuracy_sum = []
        val_accuracy_sum = []

        if __name__ == '__main__':
            # num_processes = 4
            for seed in range(n_seeds):

                trainingEpoch_loss = []
                validationEpoch_loss = []
                train_accuracy = []
                val_accuracy = []

                torch.manual_seed(base_seed[seed])
                # print(torch.manual_seed(base_seed[seed]))

                model = HybridModel()
                initialization(seed)
                params_i = []
                params_o_epoch = []
                [params_i.append(parameter) for parameter in model.parameters()]

                trainloader, testloader, dataset, testset = data_loading(data_source, batch_size, size)
                print(len(dataset))
                for epoch in range(n_epochs):
                    print(f"Tr_S: {size} Batch: {batch_size} Seed: {seed}/{n_seeds} & Epoch {epoch + 1}/{n_epochs}")
                    params_o = []
                    #   ###Training###
                    model.train()
                    train_loader = DataLoader(dataset=dataset, batch_size=batch_size)  # , num_workers=4)
                    # for param in model.parameters():
                    #    print(param.requires_grad)
                    mean_step_loss, mean_train_acc = train(model, train_loader)

                    trainingEpoch_loss.append(mean_step_loss)
                    train_accuracy.append(mean_train_acc)

                    ###Testing
                    model.eval()

                    test_loader = DataLoader(dataset=testset, batch_size=batch_size)

                    mean_val_step_loss, mean_val_train_acc = test(model, test_loader)

                    validationEpoch_loss.append(mean_val_step_loss)
                    val_accuracy.append(mean_val_train_acc)

                    # torch.save(model.state_dict(), f'{directory}model_save_{seed}_{bs}.pth')

                    defs.tac()

                    [params_o.append(parameter) for parameter in model.parameters()]

                #params_o_epoch.append(params_o)

                # [trainingEpoch_loss_sum.append(trainingEpoch_loss.to('cpu')) for i in trainingEpoch_loss]
                # [validationEpoch_loss_sum.append(validationEpoch_loss.to('cpu')) for i in validationEpoch_loss]
                # [train_accuracy_sum.append(train_accuracy.to('cpu')) for i in train_accuracy]
                # [val_accuracy_sum.append(val_accuracy.to('cpu')) for i in val_accuracy]

                # plotting_training_cycle(n_epochs)
                trainingEpoch_loss_sum.append(trainingEpoch_loss)
                validationEpoch_loss_sum.append(validationEpoch_loss)
                train_accuracy_sum.append(train_accuracy)
                val_accuracy_sum.append(val_accuracy)

                plot_data = np.array([trainingEpoch_loss,
                                      validationEpoch_loss,
                                      train_accuracy,
                                      val_accuracy])

                plot_data = np.array(plot_data)
                # torch.save(params_i, f"{directory}{size}_param_in_save_{identity}_{bs}_block{seed}.pt")
                # torch.save(params_o_epoch, f"{directory}{size}_param_out_save_{identity}_{bs}_block{seed}.pt")
                # np.save(f"{directory}{size}_Plot_data_raw_{identity}_{bs}_block{seed}.npy", plot_data)

                parameter_in.append(params_i)
                parameter_out.append(params_o)

                print(plot_data.shape)

            # p_in = []
            # p_out = []

            # for i in range(0, n_seeds):
            #     parameter_in.append(torch.load(f"{directory}{size}_param_in_save_{identity}_{bs}_block{i}.pt"))
            #     parameter_out.append(torch.load(f"{directory}{size}_param_out_save_{identity}_{bs}_block{i}.pt"))
            #
            #     dat_i = np.load(f"{directory}{size}_Plot_data_raw_{identity}_{bs}_block{i}.npy")
            #     trainingEpoch_loss_sum.append(dat_i[0])
            #     validationEpoch_loss_sum.append(dat_i[1])
            #     train_accuracy_sum.append(dat_i[2])
            #     val_accuracy_sum.append(dat_i[3])

            plot_data_sum = np.array([trainingEpoch_loss_sum,
                                      validationEpoch_loss_sum,
                                      train_accuracy_sum,
                                      val_accuracy_sum])

            print(plot_data_sum.shape)

            plot_data_sum = np.array(plot_data_sum)
            torch.save(parameter_in, f"{directory}{size}_param_in_sum_save_{identity}_{bs}.pt")
            torch.save(parameter_out, f"{directory}{size}_param_out_sum_save_{identity}_{bs}.pt")
            np.save(f"{directory}{size}_Plot_data_raw_{identity}_{bs}.npy", plot_data_sum)

        c = np.load(f"{directory}{size}_Plot_data_raw_{identity}_{bs}.npy")

        # print(c.shape)

        trainingEpoch_loss_sum = c[0]
        validationEpoch_loss_sum = c[1]
        train_accuracy_sum = c[2]
        val_accuracy_sum = c[3]

        overall_training_loss = []
        overall_validation_loss = []
        overall_training_acc = []
        overall_validation_acc = []

        overall_training_loss_avg = []
        overall_training_loss_std = []
        overall_training_loss_min = []
        overall_training_loss_max = []

        overall_validation_loss_avg = []
        overall_validation_loss_std = []
        overall_validation_loss_min = []
        overall_validation_loss_max = []

        overall_training_acc_avg = []
        overall_training_acc_std = []
        overall_training_acc_min = []
        overall_training_acc_max = []

        overall_validation_acc_avg = []
        overall_validation_acc_std = []
        overall_validation_acc_min = []
        overall_validation_acc_max = []

        for i in range(n_epochs):
            # overall_training_loss = []
            # overall_validation_loss = []
            # overall_training_acc = []
            # overall_validation_acc = []

            overall_training_loss = trainingEpoch_loss_sum[:, [i]].reshape(-1)
            overall_validation_loss = validationEpoch_loss_sum[:, [i]].reshape(-1)
            overall_training_acc = train_accuracy_sum[:, [i]].reshape(-1)
            overall_validation_acc = val_accuracy_sum[:, [i]].reshape(-1)

            # for j in range(5):
            #    overall_training_loss.append(trainingEpoch_loss_sum[j][i])
            #    overall_validation_loss.append(validationEpoch_loss_sum[j][i])
            #    overall_training_acc.append(train_accuracy_sum[j][i])
            #    overall_validation_acc.append(val_accuracy_sum[j][i])

            overall_training_loss_avg.append(sum(overall_training_loss) / len(overall_training_loss))
            overall_training_loss_std.append(np.std(overall_training_loss))
            overall_training_loss_min.append(min(overall_training_loss))
            overall_training_loss_max.append(max(overall_training_loss))

            overall_validation_loss_avg.append(sum(overall_validation_loss) / len(overall_validation_loss))
            overall_validation_loss_std.append(np.std(overall_validation_loss))
            overall_validation_loss_min.append(min(overall_validation_loss))
            overall_validation_loss_max.append(max(overall_validation_loss))

            overall_training_acc_avg.append(sum(overall_training_acc) / len(overall_training_acc))
            overall_training_acc_std.append(np.std(overall_training_acc))
            overall_training_acc_min.append(min(overall_training_acc))
            overall_training_acc_max.append(max(overall_training_acc))

            overall_validation_acc_avg.append(sum(overall_validation_acc) / len(overall_validation_acc))
            overall_validation_acc_std.append(np.std(overall_validation_acc))
            overall_validation_acc_min.append(min(overall_validation_acc))
            overall_validation_acc_max.append(max(overall_validation_acc))

        # plot_data_overall = [[[overall_training_loss_avg, overall_training_loss_min, overall_training_loss_max],
        #                      [overall_validation_loss_avg, overall_validation_loss_min, overall_validation_loss_max]],
        #                      [[overall_training_acc_avg, overall_training_acc_min, overall_training_acc_max],
        #                      [overall_validation_acc_avg, overall_validation_acc_min, overall_validation_acc_max]]]

        dif = sum(overall_validation_loss_avg) / len(overall_validation_loss_avg) - sum(overall_training_loss_avg) / len(
            overall_training_loss_avg)

        plot_data_overall = np.array([[[overall_training_loss_avg, overall_training_loss_std],
                                       [overall_validation_loss_avg, overall_validation_loss_std]],
                                      [[overall_training_acc_avg, overall_training_acc_std],
                                       [overall_validation_acc_avg, overall_validation_acc_std]],
                                      [dif]], dtype=object, )

        # df2 = pd.DataFrame(plot_data_overall)
        # df2.to_csv(f"/home/faierr/projects/qml-cybrex/models/angenc_basent/tr_sz_{size}/{size}_Plot_data_overall.csv")
        # df2.to_csv(f"/home/faierr/data/angenc_basent/{size}_Plot_data_overall.csv")

        np.save(f"{directory}{size}_Plot_data_overall_{identity}_{bs}.npy",
                plot_data_overall)

        x = np.arange(0, n_epochs, 1)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        f.suptitle(f'{size} Circuit: {identity}-{ent1}-{bs}')
        ax1.fill_between(x, np.array(overall_training_loss_avg) - np.array(overall_training_loss_std),
                         np.array(overall_training_loss_avg) + np.array(overall_training_loss_std), alpha=0.2, color='blue')
        ax1.fill_between(x, np.array(overall_validation_loss_avg) - np.array(overall_validation_loss_std),
                         np.array(overall_validation_loss_avg) + np.array(overall_validation_loss_std), alpha=0.2,
                         color='orange')
        ax1.plot(x, overall_training_loss_avg, label='train_loss_avg', color='blue')
        ax1.plot(x, overall_validation_loss_avg, label='val_loss_avg', color='orange')
        # ax1.plot(x, overall_validation_loss_min, label='val_loss_min', color = 'orange')
        # ax1.plot(x, overall_validation_loss_max, label='val_loss_max', color = 'orange')
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.set_xlim(-1, 21)
        # ax1.set_ylim(1.75,2.4)
        ax1.legend(loc='upper right')

        ax2.fill_between(x, np.array(overall_training_acc_avg) - np.array(overall_training_acc_std),
                         np.array(overall_training_acc_avg) + np.array(overall_training_acc_std), alpha=0.2, color='blue')
        ax2.fill_between(x, np.array(overall_validation_acc_avg) - np.array(overall_validation_acc_std),
                         np.array(overall_validation_acc_avg) + np.array(overall_validation_acc_std), alpha=0.2,
                         color='orange')
        ax2.plot(overall_training_acc_avg, label='train_acc')
        ax2.plot(overall_validation_acc_avg, label='val_acc')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('epoch')
        ax2.set_xlim(-1, 21)
        # ax2.set_ylim(0.6, 0.80)
        ax2.legend(loc='lower right')

        generalization_error = []

        # dif1000 = sum(overall_validation_loss_avg)/len(overall_validation_loss_avg) - sum(overall_training_loss_avg)/len(overall_training_loss_avg)
        # generalization_error.append(dif)
        # ax3.plot(generalization_error)

        f.savefig(f"{directory}{size}_plot_{identity}_{bs}.png")
        # f.tight_layout()
        # f.show()