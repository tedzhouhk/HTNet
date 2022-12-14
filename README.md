# HTNet: Dynamic WLAN Performance Prediction using Heterogenous Temporal GNN

# Dependencies

Older versions may also work but are not tested.
* python >= 3.9.7
* numpy >= 1.20.3
* pytorch >= 1.10.2
* dgl >= 0.8.0
* xgboost >= 1.5.2
* scikit-learn >= 0.24.2

# Dataset

All six setups can be downloaded on this [Google Drive Link](https://drive.google.com/file/d/1WYEVhmN4RgpLh2tGQ3A-LBQIKHjIO_q5/view?usp=sharing). After downloading, unzip the file and copy them to */data/* folder. The file structure should be 

```
├── data
│   ├── setup1
│   │   ├── processed
│   │   │   ├── train_0.bin
│   │   │   ├── valid_0.bin
│   │   │   ├── test_0.bin
│   │   │   ├── ...
│   ├── ...
├── train.py
├── ...
```

# Run

To run HTNet and the baseline methods, first specify a *setup* in *{setup1, setup2, setup3, setup4, setup5, setup6}*. Ramon, ATARI, Ramon+LSTM, ATARI+LSTM, and HTNet require to train on GPU. For these methods, specify which GPU to use using the *--gpu* option where *gpu 0* is the default value.

For SINR:
```
python train.py --data <setup> --sinr
```

For GBRT:
```
python train.py --data <setup> --gbrt
```

For Ramon:
```
python train.py --data <setup> 
```

For ATARI:
```
python train.py --data <setup> --graph
```

For Ramon+LSTM:
```
python train.py --data <setup> --dynamic
```

For ATARI+LSTM:
```
python train.py --data <setup> --graph --dynamic
```

For HTNet:
```
python train.py --data <setup> --graph --hetero --dynamic 
```