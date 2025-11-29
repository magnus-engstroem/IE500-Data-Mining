# Code for Team 9s Data Mining project

Contains data as well as the methods employed as eiter Jupyter notebooks or as .py files

### Repo structure:

```
├── rebalance_train_set.py
├── Full_data
├── Selected_data
├── SVM
│   ├── Grid_search
│   │   └── results
│   ├── PCAs
│   │   └── results
│   └── Ratios
└── Trees
```

- `rebalance_train_set.py` Helper function for up- and downsampling, used in a few different scripts in this repo
- `Full_data` Unprocessed data as downloaded
- `Selected_data` Folder containing preprocessing as well as finished data sets (`train.csv` and `test.csv`)
- `SVM` containing different tuning and evaluation scripts for SVMs
- `Trees` tuning and evaluation scripts for Decision trees and random forests