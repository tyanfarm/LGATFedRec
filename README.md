## Installation
1. Clone the repository:
``` sh
git clone https://github.com/tyanfarm/LGATFedRec
```

2. Open folder: 
``` sh
cd LGATFedRec
```

2. Install the dependencies:
``` sh
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Training settings
- Config in `train.py`
1. Movielens-100K
``` sh
--neighborhood_threshold: 0.5
--reg: 0.5
--reg_gmf: 0.25
--dataset: '100k'
```

2. Movielens-1M
``` sh
--neighborhood_threshold: 1.0
--reg: 0.5
--reg_gmf: 0.25
--dataset: 'ml-1m'
```

3. Lastfm-2K
``` sh
--neighborhood_threshold: 2.0
--reg: 0.75
--reg_gmf: 0.25
--dataset: 'lastfm-2k'
```

4. HetRec2011
``` sh
--neighborhood_threshold: 1.0
--reg: 0.5
--reg_gmf: 0.25
--dataset: 'hetrec'
```

## Start training
To start training, use the following command:

``` sh
python train.py
```


