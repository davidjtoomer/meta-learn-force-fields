# Meta-Learning Force Fields

Learning atomistic neural network potentials (NNP) using meta-learning. Final project for Stanford's [CS330: Deep Multi-Task and Meta-Learning](https://cs330.stanford.edu/).

## Environment

You can use [conda](https://docs.conda.io/en/latest/) or [pip](https://pypi.org/project/pip/) to run the code in this repository. For conda, run the following commands:

```
conda env create -f environment.yml
conda activate meta-learn-force-fields
```

For pip, first create and activate a virtual environment:

```
python -m venv venv/
source venv/bin/activate
```

and then install all of the requirements:

```
pip install -r requirements.txt
```

## Data

This project uses the [ANI-1](https://chemrxiv.org/engage/chemrxiv/article-details/60c74aabbdbb896e2ba3940c) datasets. You can download the data by running:

```
python download.py
```

## Preprocessing

We provide a script for preprocessing the ANI-1 dataset file into ANI-1x and ANI-1ccx, where each dataset is cleaned of NaN values and extraneous data. To preprocess the data, run:

```
python preprocess.py
```

This will generate two [HDF5 files](https://docs.h5py.org/en/stable/index.html), one for ANI-1x and another for ANI-1ccx. Each HDF5 file is structured with molecule names as [groups](https://docs.h5py.org/en/stable/high/group.html) at the top level. Each group contains [datasets](https://docs.h5py.org/en/stable/high/dataset.html) with the attributes <code>atomic_numbers</code>, <code>coordinates</code>, and <code>energy</code>. You can access the data for a specific molecule in your code as follows:

```python
import h5py
import numpy as np

...

dataset = '1x'
molecule_name = 'C1H1N1'
with h5py.File(f'data/ani{dataset}.h5') as f:
    molecule = f[molecule_name]
    atomic_numbers = np.array(molecule['atomic_numbers'])
    coordinates = np.array(molecule['coordinates'])
    energy = np.array(molecule['energy'])

...
```
