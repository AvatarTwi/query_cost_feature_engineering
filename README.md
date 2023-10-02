# QCFE: An efficient Feature engineering for query cost estimation.

QCFE is a feature engineering to improve the time-accuracy efficiency for query cost estimation. This is the implementation described in the paper: 

QCFE: An efficient Feature engineering for query cost estimation.

![qcfe](https://typora-picpool-1314405309.cos.ap-nanjing.myqcloud.com/img/qcfe.png)

## Source Code Structure

- `build_template/*`: build template queries for efficiently estimating the feature snapshot
- `dataset/`
  - `postgres_tpch_dataset/*`
    - `snapshot/`: calculate feature snapshot
    - `attr_val_dict.pickle`: including the min, max, and med values of each attribute in TPCH
    - `tpch_utils_knob.py`: generate a labeled dataset with feature snapshot for QPPNet
    - `tpch_utils_origin.py`: generate a labeled dataset without feature snapshot for QPPNet
    - `tpch_utils_serialize_knob.py`: generate a labeled dataset with feature snapshot for MSCN
    - `tpch_utils_serialize.py`: generate a labeled dataset without feature snapshot for MSCN
  - ... : other workloads, similar to `postgres_tpch_dataset`
- `greedy/`: the approximate greedy feature reduction algorithm
- `models/`: QCFE(qpp), QCFE(mscn), [QPPNet](https://github.com/rabbit721/QPPNet), and [MSCN](https://github.com/andreaskipf/learnedcardinalities).
- `build_dataset.py`: build dataset
- `build_model.py`: build the model in different modes
- `config.py`: global variables 
- `main.py`: the main script used for running experiments

## Environment

Tested with python 3.10. To install the required packages, please run the following command:

```shell
conda install --yes --file requirements.txt
```

## Run Experiments

The main script used for running experiments is `main.py`. It can be invoked using the following syntax:

```shell
python main.py --scale <scale> --benchmark <benchmark> --model <model> --type <type>
```

where `<scale>` is the sampling scale of labeled data, and `<benchmark>` is the benchmark used in the experiment, which must be TPCH, Sysbench, or job-light. `<model>` is QPPNet or MSCN. `<type>` is the type of experiment.

## Usage Examples

### QCFE vs Origin

To run QPPNet in TPCH with scale=2000, one should run the following command: 

```shell
python main.py --scale 2000 --workload TPCH --model QPPNet --type origin_model
python main.py --scale 2000 --workload Sysbench --model QPPNet --type origin_model
python main.py --scale 2000 --workload job-light --model QPPNet --type origin_model
```

To run the QCFE in TPCH with scale=2000, one should run the following command:

```shell
python main.py --scale 2000 --workload TPCH --model QPPNet --type snapshot_model_FR
python main.py --scale 2000 --workload Sysbench --model QPPNet --type snapshot_model_FR
python main.py --scale 2000 --workload job-light --model QPPNet --type snapshot_model_FR
```

Similarly, one can run the other model:

```shell
python main.py --scale 2000 --workload TPCH --model MSCN --type snapshot_model_FR
python main.py --scale 2000 --workload Sysbench --model MSCN --type snapshot_model_FR
python main.py --scale 2000 --workload job-light --model MSCN --type snapshot_model_FR
```

### Template

The `<scale>` is used to set the scale of template queries

```shell
python main.py --scale 4000 --workload TPCH --model QPPNet --type snapshot_modeltemplate1
python main.py --scale 4000 --workload TPCH --model QPPNet --type snapshot_modeltemplate2
python main.py --scale 4000 --workload TPCH --model QPPNet --type snapshot_modeltemplate3
python main.py --scale 4000 --workload TPCH --model QPPNet --type snapshot_modeltemplate4
```

## Reference

If you find this repository useful in your work, please cite our paper:

```

```

