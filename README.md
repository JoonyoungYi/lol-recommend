# lol-recommend
* LoL Champion Recommendation

## Background
* This repository is for recommending the champion's of the LoL game.
* This repository is slightly changed from [this repository](https://github.com/JoonyoungYi/BiasedMF-tensorflow).


## Environment
* I've tested this repository on Python 3.5 and Ubuntu 16.04.

* Before start:
  * Prepare `win_lose_data.npy` in the root directory of this repository.
  * You could get this data file from `신현석`.

* How to init:
```
$ virtualenv .venv -p python3
$ . .venv/bin/activate
$ pip install -r requirements.txt
$ mkdir logs
$ deactivate
```

* How to run:
```
$ . .venv/bin/activate
$ python run.py
```
