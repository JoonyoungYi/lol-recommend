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

| RANK | TRAINING RMSE | TEST RMSE  |
|------|---------------|------------|
|   2  |               |            |
