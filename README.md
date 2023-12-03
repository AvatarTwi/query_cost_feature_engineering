# QCFE: An efficient Feature engineering for query cost estimation.

QCFE is a feature engineering to improve the time-accuracy efficiency for query cost estimation. This is the implementation described in the paper: 

QCFE: An efficient Feature engineering for query cost estimation.[PDF](https://arxiv.org/pdf/2310.00877.pdf)

![qcfe1](https://typora-picpool-1314405309.cos.ap-nanjing.myqcloud.com/img/qcfe1.png)

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

### Data Preparation

Our training data and testing data can be downloaded in the [link](https://drive.google.com/file/d/1iSzXmHDcSgeDRACWgTjdjBFMACsnCXAG/view?usp=sharing). Unzip it in the root dir of the code.

### Main Script

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

## Reference

If you find this repository useful in your work, please cite our paper:

```
@misc{yan2023qcfe,
      title={QCFE: An efficient Feature engineering for query cost estimation}, 
      author={Yu Yan and Hongzhi Wang and Junfang Huang and Dake Zhong and Man Yang and Kaixin Zhang and Tao Yu and Tianqing Wan},
      year={2023},
      eprint={2310.00877},
      archivePrefix={arXiv},
      primaryClass={cs.DB}
}
```

## Code Citations

We utilize some open source libraries to implement our experiments. The specific citation is as follows:

```
Github repository: SHAP. https://github.com/shap/shap.
Github repository: QPPNet in PyTorch. https://github.com/rabbit721/QPPNet.
Github repository: MSCN. https: //github.com/andreaskipf/learnedcardinalities.
```



