# QCFE: An efficient Feature engineering for query cost estimation.

QCFE is a feature engineering for query cost estimation to improve the time-accuracy efficiency. This is the implementation described in the paper: QCFE: An efficient Feature engineering for query cost estimation.

![qcfe](https://typora-picpool-1314405309.cos.ap-nanjing.myqcloud.com/img/qcfe.png)

## Source Code Structure

- `build_template/*`: build template queries of TPCH and job-light
- `dataset/`
  - `postgres_tpch_dataset/*`: 
    - `cost_factor/`: calculate snapshot
    - `attr_rel_dict.py`: the data abstract of TPCH
    - `attr_val_dict.pickle`: including the min, max, and med values of each attribute in TPCH
    - `tpch_utils_knob.py`: generate a dataset with feature snapshot for QPPNet
    - `tpch_utils_origin.py`: generate a dataset without feature snapshot for QPPNet
    - `tpch_utils_serialize_knob.py`: generate a dataset with feature snapshot for MSCN
    - `tpch_utils_serialize.py`: generate a dataset without feature snapshot for MSCN
  - ... : other workloads, similar to `postgres_tpch_dataset`
- `greedy/`: define the greedy method in feature reduction 
- `models/`: define two models, QPPNet and MSCN.
- `utils/`: utils for experiments
- `build_dataset.py`: build dataset
- `build_model.py`: build the model in different modes
- `config.py`: define global variables 
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

where `<scale>` is the sampling scale of labeled data, and `<benchmark>` is the benchmark used in the experiment, which must be TPCH, Sysbench, or job-light. `<model>` is qppnet or mscn. `<type>` is the type of experiment.

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
python main.py --scale 2000 --workload TPCH --model QPPNet --type qcfe
python main.py --scale 2000 --workload Sysbench --model QPPNet --type qcfe
python main.py --scale 2000 --workload job-light --model QPPNet --type qcfe
```

Similarly, one can run the other model:

```shell
python main.py --scale 2000 --workload TPCH --model MSCN --type qcfe
python main.py --scale 2000 --workload Sysbench --model MSCN --type qcfe
python main.py --scale 2000 --workload job-light --model MSCN --type qcfe
```

### Feature Reduction Experiment

three kinds of feature reduction: shap, grad, greedy

```shell
python main.py --scale 2000 --workload TPCH --model QPPNet --type knob_model_shap
python main.py --scale 2000 --workload TPCH --model QPPNet --type knob_model_grad
python main.py --scale 2000 --workload TPCH --model QPPNet --type knob_model_greedy
```

### Template Experiment

Feature snapshot is generated by template data. `<num>` is the number of templates. 

```shell
python main.py --scale 2000 --workload TPCH --model QPPNet --type knob_modeltemplate_<num>
```

### Transfer Experiment

`knob_modeltransfer` is to test transferability. (`knob_modeltransfer_template`: Feature snapshot is generated by template.)

```shell
python main.py --scale 2000 --workload TPCH --model QPPNet --type knob_modeltransfer
python main.py --scale 2000 --workload TPCH --model QPPNet --type knob_modeltransfer_template
```

## How to experiment with QCFE on a new Dataset

- Specify the dataset in dataset/xxx_dataset/attr_rel_dict.py
- Due to the current implementation, make sure to declare: table names, index names, and table-attribute dictionary.
- dataset/xxx_dataset/attr_val_dict.pickle including the min, max, and med values of each attribute.

## Reference

If you find this repository useful in your work, please cite our paper:

```

```
