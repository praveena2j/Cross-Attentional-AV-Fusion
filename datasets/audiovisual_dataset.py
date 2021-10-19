import torch.utils.data as data
import sys
import torch
import numpy as np
def refined_datasets(datasets):
    ref_audiodata = []
    ref_visualdata = []
    ref_labels = []
    ref_datasets  = []
    print(len(datasets))
    print(len(datasets[0]))
    print(len(datasets[1]))

    for visdata in datasets[1]:
        print(len(visdata))
        sys.exit()


    for audiodata, visualdata in zip(datasets[0], datasets[1]):
        if torch.is_tensor(visualdata[0]):
        #if visualdata[0].shape[0] == 128:
            ref_audiodata.append(audiodata[0])
            ref_labels.append(np.mean(audiodata[1]))
            ref_visualdata.append(visualdata[0])
            ref_datasets.append([audiodata[0], visualdata[0], audiodata[1]])
    print(len(ref_datasets))
    sys.exit()
    return ref_datasets



class ConcatDataset(data.Dataset):
    def __init__(self, *datasets):
        #self.datasets = datasets #refined_datasets(datasets)
        self.datasets = refined_datasets(datasets)

    def __getitem__(self, i):
        #audiodata, visualdata, label = self.datasets[i]
        #return audiodata, visualdata, label
        #for d in self.datasets:
        #    print(d[i])
        #sys.exit()
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
        #return len(self.datasets)