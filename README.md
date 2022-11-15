# Meta-Learning Force Fields

Learning atomistic neural network potentials (NNP) using meta-learning. Final project for Stanford's [CS330: Deep Multi-Task and Meta-Learning](https://cs330.stanford.edu/).

## Data

This project uses the [ANI-1](https://chemrxiv.org/engage/chemrxiv/article-details/60c74aabbdbb896e2ba3940c) datasets. You can download the data by running:

```
python download.py
```

## Preprocessing

We provide a script for preprocessing the ANI-I dataset file into ANI-1x and ANI-1ccx, where each dataset is cleaned of NaN values. To preprocess the data, run:

```
python preprocess.py
```

Furthermore, we sort the files into a directory structure well suited for multi-task and meta-learning. Within the meta-learning framework, we consider each molecule to be a task. Individual configurations of a given molecule, along with their respective potential energies, are samples of the task. After preprocessing, we structure our data directory as

```
meta-learn-force-fields
└───data
    └───DATASET_NAME
        └───MOLECULE_NAME
            │   0.pt
            │   1.pt
            │   ...
```

Each configuration file contains the attributes <code>atomic_numbers</code>, <code>coordinates</code>, and <code>energy</code> stored as [PyTorch tensors](https://pytorch.org/docs/stable/tensors.html). You can load any one of these configurations like so:

```python
import torch

...

task = 'C1H1N1'
task_sample = torch.load(f'data/1x/{task}/0.pt')
atomic_numbers = task_sample['atomic_numbers']
coordinates = task_sample['coordinates']
energy = task_sample['energy']

...
```
