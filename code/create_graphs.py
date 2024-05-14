from resnet import (
    eval_model,
    test_dataloader,
    test_data,
    test_labels,
    get_logits,
    CIFARDataset,
    load_synth_data,
    collect_train,
    EnsClassifier,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np


def plot_ens_student_teacher_graphs():
    logits_list = []

    for i in range(12):
        teacher = torch.load(f"ensemble/ens-{i}.pth")
        print(eval_model(teacher, test_dataloader, nn.CrossEntropyLoss())[1])
        teacher_preds = get_logits(teacher, test_data)
        logits_list.append(teacher_preds)

    test_labels = torch.Tensor(test_labels).to("cuda")
    preds = logits_list[0]
    correct_list = [
        torch.eq(torch.argmax(preds, dim=1), test_labels).sum().item()
        / len(test_labels)
    ]
    for logits in logits_list[1:]:
        preds += logits
        correct_list.append(
            torch.eq(torch.argmax(preds, dim=1), test_labels).sum().item()
            / len(test_labels)
        )

    student_correct_list = []
    for i in range(1, 13):
        student = torch.load(f"ens-num-distill/Res56-{i}.pth")
        preds = get_logits(student, test_data)
        student_correct_list.append(
            torch.eq(torch.argmax(preds, dim=1), test_labels).sum().item()
            / len(test_labels)
        )

    plt.xlabel("Size of Teacher Ensemble")
    plt.ylabel("Test Accuracy")
    plt.plot(range(1, len(correct_list) + 1), correct_list, label="Teacher")
    plt.plot(range(1, len(correct_list) + 1), student_correct_list, label="Student")
    plt.legend()
    plt.savefig("ens")
    plt.close()


def plot_ens_student_teacher_agreement():
    logits_list = []

    for i in range(12):
        teacher = torch.load(f"ensemble/ens-{i}.pth")
        teacher_preds = get_logits(teacher, test_data)
        logits_list.append(teacher_preds)

    preds = None
    student_agreement_list = []

    for logits in logits_list:
        if preds is None:
            preds = logits
        preds += logits
        teacher_preds = torch.argmax(preds, dim=-1)

        student = torch.load(f"ens-num-distill/Res56-{i}.pth")
        student_preds = get_logits(student, test_data)
        student_agreement_list.append(
            torch.eq(torch.argmax(student_preds, dim=1), teacher_preds).sum().item()
            / len(teacher_preds)
        )

    plt.xlabel("Size of Teacher Ensemble")
    plt.ylabel("Test Agreement")
    plt.plot(
        range(1, len(student_agreement_list) + 1),
        student_agreement_list,
    )
    plt.savefig("ens-agg")
    plt.close()


def graph_added_data(self_distill):
    teacher = (
        torch.load("best.pth")
        if self_distill
        else EnsClassifier([torch.load(f"ensemble/ens-{i}.pth") for i in range(3)], 1)
    )
    teacher_acc = eval_model(teacher, test_dataloader, nn.CrossEntropyLoss())[1] * 100
    teacher_preds = torch.argmax(get_logits(teacher, test_data), dim=-1)

    test_agreement_dataloader = DataLoader(
        CIFARDataset(test_data, teacher_preds), batch_size=256
    )

    gans = [0, 12_500, 25_000, 37_500, 50_000]
    test_accs, test_agrees = [], []

    for gan in gans:
        model = (
            torch.load(f"self-distill/Res56-{gan}.pth")
            if self_distill
            else torch.load(f"ens-distill/Res56-{gan}.pth")
        )
        test_acc = eval_model(model, test_dataloader, nn.CrossEntropyLoss())[1] * 100
        test_agree = (
            eval_model(model, test_agreement_dataloader, nn.CrossEntropyLoss())[1] * 100
        )
        test_accs.append(test_acc)
        test_agrees.append(test_agree)

    gans = [gan / 1000 for gan in gans]
    plt.plot(gans, test_agrees, label="Test Agreement", color="blue")
    plt.plot(
        gans,
        [teacher_acc] * len(gans),
        label="Teacher Accuracy",
        linestyle="dashed",
        color="black",
    )
    plt.plot(gans, test_accs, label="Student Accuracy", color="green")
    plt.legend()
    plt.xlabel("Number of Additinal Training Images (Thousands)")
    name = "graph" if self_distill else "ens-graph"
    plt.savefig(name)
    plt.close()


def show_synth_data():
    images = load_synth_data() * 255
    num_samples = 4
    scale_factor = 2

    sample_indices = np.random.choice(len(images), size=num_samples, replace=False)
    selected_images = images[sample_indices]
    zoomed_images = [
        zoom(image, zoom=(scale_factor, scale_factor, 1), order=3)
        for image in selected_images
    ]

    _, axes = plt.subplots(nrows=1, ncols=num_samples, figsize=(12, 4))

    for i, (ax, zoomed_image) in enumerate(zip(axes.flat, zoomed_images)):
        ax.imshow(zoomed_image.astype(np.uint8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Image {sample_indices[i]}")

    plt.tight_layout()
    plt.savefig("examples")
    plt.close()


if __name__ == "__main__":
    plot_ens_student_teacher_agreement()
    # graph_added_data(False)
    # graph_added_data(True)
    # show_synth_data()
    # plot_ens_student_teacher_graphs()
