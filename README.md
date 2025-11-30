# Code for Team 9s Data Mining project

Contains data as well as the methods employed as eiter Jupyter notebooks or as .py files

### Repo structure:

```
.
├── Base_line
├── Full_data
├── Selected_data
├── SVM
│   ├── Grid_search
│   │   └── results
│   ├── PCAs
│   │   └── results
│   └── Ratios
├── Trees
└── rebalance_train_set.py


```
- `Base_line` scripts for majority class base and decision stump base line
- `Full_data` Unprocessed data as downloaded
- `Selected_data` Folder containing preprocessing, including full list of features, analysis as well as finished data sets (`train.csv` and `test.csv`)
- `SVM` containing different tuning and evaluation scripts for SVMs
- `Trees` tuning and evaluation notebooks for Decision trees and random forests
- `rebalance_train_set.py` Helper function for up- and downsampling, used in a few different scripts in this repo


In case file paths are wrong, the data sets can be accessed like this:
```python

url_train = "https://raw.githubusercontent.com/magnus-engstroem/IE500-Data-Mining/main/Selected_data/train.csv"
train  = pd.read_csv(url_train)

url_test = "https://raw.githubusercontent.com/magnus-engstroem/IE500-Data-Mining/main/Selected_data/test.csv"
test   = pd.read_csv(url_test)
```