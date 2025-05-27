## Requirements
- Python version 3.11

## Installation
Install the dependencies:
``` sh
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Start training
To start training, use the following command:

1. Movielens-100K
``` sh
python train.py --neighborhood_threshold 0.5 --reg 0.5 --reg_gmf 0.25 --dataset 100k
```

2. Movielens-1M
``` sh
python train.py --neighborhood_threshold 1.0 --reg 0.5 --reg_gmf 0.25 --dataset ml-1m
```

3. Lastfm-2K
``` sh
python train.py --neighborhood_threshold 2.0 --reg 0.75 --reg_gmf 0.25 --dataset lastfm-2k
```

4. HetRec2011
``` sh
python train.py --neighborhood_threshold 1.0 --reg 0.5 --reg_gmf 0.25 --dataset hetrec
```