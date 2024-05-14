import torch
import pickle
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import matplotlib.pyplot as plt
from model import PreResNet
from losses import KDLoss, KDLossProb
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200
temperature = 1


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def eval_model(model, test_dataloader, criterion):
    total_test_loss = 0
    test_batches = 0
    test_correct = 0
    test_total = 0
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(inputs)
            loss = criterion(preds, labels)
            total_test_loss += loss.item()
            test_batches += 1
            test_correct += torch.eq(torch.argmax(preds, dim=1), labels).sum().item()
            test_total += len(labels)

    return total_test_loss / test_batches, test_correct / test_total


def train_model(
    model,
    optimizer,
    train_dataloader,
    test_dataloader,
    name,
    EPOCHS=100,
    teaching=(False, None),
):
    cross_entropy = nn.CrossEntropyLoss()
    if teaching[0]:
        kd_loss = teaching[1](teaching[0])
    optimizer, scheduler = optimizer
    model.to(device)
    train_losses = []
    test_losses = []
    test_accs = []
    best_test_acc = 0
    for epoch in range(EPOCHS):
        total_train_loss = 0
        train_batches = 0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            if teaching:
                loss = kd_loss(preds, labels)
            else:
                loss = cross_entropy(preds, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_batches += 1
        scheduler.step()

        test_loss, test_acc = eval_model(model, test_dataloader, cross_entropy)

        train_losses.append(total_train_loss / train_batches)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        if test_accs[-1] > best_test_acc:
            torch.save(model, f"{name}-bestacc.pth")

        if epoch % 10 == 0 or epoch + 1 == EPOCHS:
            graph_losses(train_losses, test_losses, test_accs, name)

        # print(f"   {total_train_loss/train_batches}  {total_test_loss/test_batches}")

    os.makedirs(os.path.dirname(f"{name}.pth"), exist_ok=True)
    os.makedirs(os.path.dirname("plots/{name}.png"), exist_ok=True)

    torch.save(model, f"{name}.pth")
    return train_losses, test_losses, test_acc


def graph_losses(train_losses, test_losses, test_acc, name):
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, test_losses, label="test")
    plt.legend()
    plt.savefig(f"plots/{name}.png")
    plt.close()
    plt.plot(epochs, test_acc, label="test acc")
    plt.savefig(f"plots/{name}-acc.png")
    plt.close()


class EnsClassifier(nn.Module):
    def __init__(self, models, temperature):
        super().__init__()
        self.models = models
        self.temp = temperature
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        ens_probs = None
        for model in self.models:
            pred = self.softmax(model(x) / self.temp)
            if ens_probs is None:
                ens_probs = pred
            else:
                ens_probs += pred

        return ens_probs


class CIFARDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]), torch.tensor(self.labels[idx])


def get_logits(model, data, batch_size=256):
    model.to(device)
    logits = None
    dataloader = DataLoader(
        CIFARDataset(data, range(len(data))), batch_size=batch_size, shuffle=False
    )
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            preds = model(inputs)
            if logits is None:
                logits = preds
            else:
                logits = torch.cat((logits, preds), dim=0)
    return logits


class CIFARDistillation(Dataset):
    def __init__(self, data, teacher_model):
        self.data = data
        self.teacher_labels = get_logits(teacher_model, data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        return torch.Tensor(data), self.teacher_labels[idx].clone()


def collect_train():
    labels = []
    data = None
    for i in range(5):
        train = unpickle(f"cifar-10-python/cifar-10-batches-py/data_batch_{i+1}")
        labels += train[b"labels"]
        if data is None:
            data = train[b"data"]
        else:
            data = np.vstack((data, train[b"data"]))
    return data.reshape((data.shape[0], 3, 32, 32)) / 255, labels


def load_synth_data():
    data = None
    for i in range(50):
        train = np.load(f"generated/generate-cifar10-{i}.npy")
        if data is None:
            data = train
        else:
            data = np.vstack((data, train))
    return data


def train_resnet(name):
    model = PreResNet(num_classes=10, depth=56)
    optim = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    train_model(
        model, (optim, sched), train_dataloader, test_dataloader, name, EPOCHS=EPOCHS
    )


def train_ensembles():
    os.makedirs("ensemble", exist_ok=True)
    for i in range(12):
        train_resnet(f"ensemble/ens-{i}")


def train_self_distill(teacher=None):

    if teacher is None:
        train_resnet("teacher")
        teacher = "teacher.pth"
    for num_gans in [0, 12_500, 25_000, 37_500, 50_000]:
        distill_train_dataloader = DataLoader(
            CIFARDistillation(
                np.vstack((train_data, synth_data[:num_gans])),
                torch.load(teacher),
            ),
            batch_size=256,
            shuffle=True,
        )

        model = PreResNet(num_classes=10, depth=56)
        optim = SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=EPOCHS, eta_min=1e-6
        )
        train_model(
            model,
            (optim, sched),
            distill_train_dataloader,
            test_dataloader,
            f"self-distill/Res56-{num_gans}",
            EPOCHS=EPOCHS,
            teaching=(temperature, KDLoss),
        )


def train_3ens_distill_gan():
    for num_gans in [0, 12_500, 25_000, 37_500, 50_000]:
        distill_train_dataloader = DataLoader(
            CIFARDistillation(
                np.vstack((train_data, synth_data[:num_gans])),
                EnsClassifier(
                    [torch.load(f"ensemble/ens-{i}.pth") for i in range(3)], temperature
                ),
            ),
            batch_size=256,
            shuffle=True,
        )

        model = PreResNet(num_classes=10, depth=56)
        optim = SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=EPOCHS, eta_min=1e-6
        )
        train_model(
            model,
            (optim, sched),
            distill_train_dataloader,
            test_dataloader,
            f"ens-distill/Res56-{num_gans}",
            EPOCHS=EPOCHS,
            teaching=(temperature, KDLoss),
        )


def ens_num_distill():
    for ens_size in range(1, 13):
        distill_train_dataloader = DataLoader(
            CIFARDistillation(
                train_data,
                EnsClassifier(
                    [torch.load(f"ensemble/ens-{i}.pth") for i in range(ens_size)],
                    temperature,
                ),
            ),
            batch_size=256,
            shuffle=True,
        )

        model = PreResNet(num_classes=10, depth=56)
        optim = SGD(model.parameters(), lr=5e-2, momentum=0.9, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=EPOCHS, eta_min=1e-6
        )
        train_model(
            model,
            (optim, sched),
            distill_train_dataloader,
            test_dataloader,
            f"ens-num-distill/Res56-{ens_size}",
            EPOCHS=EPOCHS,
            teaching=(temperature, KDLossProb),
        )


meta = unpickle("cifar-10-python/cifar-10-batches-py/batches.meta")
test = unpickle("cifar-10-python/cifar-10-batches-py/test_batch")

train_data, train_labels = collect_train()
test_data, test_labels = test[b"data"], test[b"labels"]
test_data = test_data.reshape((test_data.shape[0], 3, 32, 32)) / 255

synth_data = load_synth_data().transpose((0, 3, 1, 2))

train_dataloader = DataLoader(
    CIFARDataset(train_data, train_labels), batch_size=256, shuffle=True
)
test_dataloader = DataLoader(CIFARDataset(test_data, test_labels), batch_size=256)

if __name__ == "__main__":

    train_ensembles()
    train_self_distill("ensemble/ens-0.pth")
    train_3ens_distill_gan()
    ens_num_distill()
