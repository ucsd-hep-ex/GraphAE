import torch
from itertools import chain
from util.scaler import Standardizer
from datagen.graph_data_gae import GraphDataset
from sklearn.preprocessing import StandardScaler

def test_standardization(data):
    scaler1 = Standardizer()
    scaler2 = StandardScaler()
    scaler1.fit(data)
    scaler2.fit(data.numpy())
    t1 = scaler1.transform(data)
    t2 = torch.tensor(scaler2.transform(data),dtype=torch.float32)
    assert torch.allclose(t1, t2, atol=1e-7), "Custom scaler not matching sklearn"
    assert not torch.allclose(t1, data, atol=1e-7), "Did not transform data"

    t3 = scaler1.inverse_transform(t1)
    t4 = torch.tensor(scaler2.inverse_transform(t2.numpy()),dtype=torch.float32)

    assert torch.allclose(t1, t2, atol=1e-7), "Inverse transformations do not match"
    return "standardization good"

if __name__ == '__main__':
    gdata = GraphDataset(root='/anomalyvol/data/bb_train_sets/test_rel/', bb=0)
    gdata = [data for data in chain.from_iterable(gdata)]
    x = torch.cat([d.x for d in gdata])
    print(test_standardization(x))
