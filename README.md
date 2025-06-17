## Requirements
- Python version 3.10
- OS: Linux (e.g. Ubuntu 20.04)

## Installation
Install the dependencies:
``` sh
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Training Code
- Implemented in `engine.py`

## Evaluation Code
- Implemented in `metrics.py`

## Results and Commands
| Dataset        | HR\@10 | NDCG\@10 | Command to Run                                                                                |
| -------------- | ------ | -------- | --------------------------------------------------------------------------------------------- |
| MovieLens-100K | 93.96  | 82.63    | `python train.py --neighborhood_threshold 1.5 --reg 0.75 --reg_gmf 0.25 --dataset 100k`        |
| MovieLens-1M   | 89.64  | 70.81    | `python train.py --neighborhood_threshold 1.0 --reg 0.5 --reg_gmf 0.25 --dataset ml-1m`       |
| Lastfm-2K      | 84.31  | 74.54    | `python train.py --neighborhood_threshold 2.25 --reg 0.75 --reg_gmf 0.25 --dataset lastfm-2k` |
| HetRec2011     | 86.46  | 72.51    | `python train.py --neighborhood_threshold 1.5 --reg 0.75 --reg_gmf 0.25 --dataset hetrec`     |
