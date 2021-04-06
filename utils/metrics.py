import tqdm
import numpy as np
import torch

def compute_ACSA_on_dataset(model, dataset, nb_classes=2, ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with tqdm.tqdm(enumerate(dataset, 0), desc='Computing ASCA', total=len(dataset)) as test_pbar:
        for i, data in test_pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # forward + preds
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, dim=1)

            # update confusion matrix
            for label, pred in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[label.long(), pred.long()] += 1

    return get_ACSA(confusion_matrix)

def compute_ACSA_on_tensors(preds, labels) -> list:
    confusion_matrix = get_confusion_matrix(preds, labels)
    return confusion_matrix.diag()/confusion_matrix.sum(1)

def get_CSA(confusion_matrix) -> list:
    return confusion_matrix.diag()/confusion_matrix.sum(1)

def get_confusion_matrix(preds, labels):
    # nb_classes = len(np.unique(labels))
    nb_classes = len(np.unique(labels.cpu().numpy()))
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    for label, pred in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[label.long(), pred.long()] += 1
    return confusion_matrix