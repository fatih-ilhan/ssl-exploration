# ssl-exploration
This repository contains the code for:

F. Ilhan and E. Mumcuoglu, “Performance Analysis of Semi-Supervised Learning Methods under Different Missing Label Patterns”, 28th IEEE Signal Processing and Communications Applications, 2020.

## Run the code:
To run the experiments, give the following arguments to main.py:

```python
parser.add_argument('--mode', type=str, nargs='+')  # "train", "test"
parser.add_argument('--model_name', type=str)  # "self_trainer", "s3vm", "gmm"
parser.add_argument('--dataset_list', type=str, nargs='+')  # string list
parser.add_argument('--num_repeat', type=int, default=1)  # repeat train + test
parser.add_argument('--save', type=bool, default=1)  # save results flag
parser.add_argument('--load', type=bool, default=0)  # load previous pkl
```
You can set model, data and simulation parameter loops from `config.py`.
