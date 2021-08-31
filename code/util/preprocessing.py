import torch
from sklearn.preprocessing import StandardScaler

def get_iqr_proportions(dataset):
    """
    calcualte iqr proportions of training dataset for weighted mse

    :param dataset: list of Data objects
    """
    all_x = torch.cat([d.x for d in dataset])
    qs = torch.quantile(all_x, torch.tensor([0.25,0.75]), dim=0)
    iqr = qs[1] - qs[0]
    max_iqr = torch.max(iqr)
    iqr_prop = iqr / max_iqr
    return iqr_prop


def standardize(train_dataset, valid_dataset, test_dataset):
    """
    standardize dataset and return scaler for inversion

    :param train_dataset: list of Data objects
    :param valid_dataset: list of Data objects
    :param test_dataset: list of Data objects

    :return scaler: sklearn StandardScaler
    """
    train_x = torch.cat([d.x for d in train_dataset]).numpy()

    scaler = StandardScaler()
    scaler.fit(train_x)
    for dataset in (train_dataset, valid_dataset, test_dataset):
        for d in dataset:
            d.x[:,:] = torch.from_numpy(scaler.transform(d.x.numpy()))
    return scaler
