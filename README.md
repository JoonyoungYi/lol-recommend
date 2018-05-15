# lol-recommend
* LoL Champion Recommendation

## Background
* This repository is for recommending the champion's of the LoL game.
* This repository is slightly changed from [this repository](https://github.com/JoonyoungYi/MCAM-numpy).
* The theoretical background is from the paper, [Low-rank Matrix Completion using Alternating Minimization](https://arxiv.org/abs/1212.0467).
  * I removed the process of splitting omega in this repository.
  * I removed the clipping process in initializing U in this repository.
* I represent `Omega` as `mask` in this repository.
  * mask[i][j] =  1 if entry (i, j) is training case
  * mask[i][j] = -1 if entry (i, j) is test case
  * mask[i][j] =  0 otherwise

## Environment
* I've tested this repository on Python3.6 and OS X(10.13.4).

* Before start:
  * Prepare `win_lose_data.npy` in the root directory of this repository.
  * You could get this data file from `신현석`.

* How to init:
```
$ virtualenv .venv -p python3.6
$ . .venv/bin/activate
$ pip install -r requirements.txt
$ deactivate
```

* How to run:
```
$ . .venv/bin/activate
$ python run.py
```



## Performance
* `training case : test case = 9 : 1`

### Base Line Algorithm
* MEAN: 0.5257707664608374
* RMSE: 0.12961570956955257

### Biased MF
| RANK | TRAINING RMSE | TEST RMSE |
|------|---------------|-----------|
| 2 | 0.3592192670994933 | 0.3974507102907396 |
| 3 | 0.3497304224904734 | 0.4075213813496369 |
| 4 | 0.3406751669348579 | 0.41786657128441657 |
| 5 | 0.33125098894410804 | 0.4274329229101035 |
| 6 | 0.3218559021503986 | 0.43698928194311903 |
| 7 | 0.3131985664090909 | 0.44563588552818506 |
