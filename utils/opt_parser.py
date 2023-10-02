import argparse

import config


def getParser(version, dataset, new_ds, new_md, mid_data_dir, data_structure, data_dir, saved_model, mode,
              knobs="300-300-300",change=False,start_epoch=0,end_epoch=400,batch_size=1024,scale=2000):

    parser = argparse.ArgumentParser(description='QPPNet Arg Parser')

    parser.add_argument('--new_data_structure', action='store_true', default=new_ds, help='new datastruct or no')

    parser.add_argument('--new_mid_data', action='store_true', default=new_md, help='new mid data or no')

    parser.add_argument('--change', action='store_true', default=change, help='change or no')

    parser.add_argument('--data_structure', type=str, default=data_structure,
                        help='data structure path')

    parser.add_argument('--mid_data_dir', type=str, default=mid_data_dir)

    parser.add_argument('--knobs', type=str, default=knobs)

    parser.add_argument('--version', type=str, default=version)

    parser.add_argument('--mode', type=str, default=mode, help='train/test/part_train')

    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch size used in training (default: 128)')

    parser.add_argument('--data_dir', type=str, default=data_dir, help='Dir containing collected data')

    parser.add_argument('--dataset', type=str, default=dataset,
                        help='Select dataset [PSQLTPCH | TerrierTPCH | OLTP]')

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')

    parser.add_argument('--scheduler', action='store_true', default=True)

    parser.add_argument('--step_size', type=int, default=1000,
                        help='step_size for StepLR scheduler (default: 1000)')

    parser.add_argument('--gamma', type=float, default=0.95,
                        help='gamma in Adam (default: 0.95)')

    parser.add_argument('--SGD', action='store_true',default=False,
                        help='Use SGD as optimizer with momentum 0.9')

    parser.add_argument('-s', '--start_epoch', type=int, default=start_epoch,
                        help='Epoch to start training with (default: 0)')

    parser.add_argument('-t', '--end_epoch', type=int, default=end_epoch,
                        help='Epoch to end training (default: 200)')

    parser.add_argument('-epoch_freq', '--save_latest_epoch_freq', type=int, default=100)

    parser.add_argument('-dir', '--save_dir', type=str, default=saved_model,
                        help='Dir to save model weights (default: ./saved_model)')

    parser.add_argument('-logf', '--logfile', type=str, default='train_loss.txt')

    parser.add_argument('--scale', type=int, default=scale)

    parser.add_argument('--benchmark', type=str, default='tpch')

    parser.add_argument('--model', type=str, default='QPPNet')

    parser.add_argument('--type', type=str, default='origin_model')


    return parser


def defaultParser():

    parser = argparse.ArgumentParser(description='QPPNet Arg Parser')

    # 2000,4000,6000,8000,10000
    parser.add_argument('--scale', type=int, default=2000, help='data scale')

    # 'TPCH','Sysbench','job-light'
    parser.add_argument('--workload', type=str, default='job-light', help='benchmark')

    # "QPPNet","MSCN"
    parser.add_argument('--model', type=str, default='QPPNet', help='qppnet or mscn')

    # "origin_model",
    # "snapshot_model",
    # "snapshot_modeltransfer",
    # "snapshot_modeltransfer_template",
    # "snapshot_model_FR",
    # "snapshot_model_GD",
    # 'snapshot_model_GREEDY',
    # 'snapshot_modeltemplate' + str(scale),
    parser.add_argument('--type', type=str, default='origin_model', help='mid dir path')

    return parser


def save_opt(opt, logf):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)
    logf.write(message)
    logf.write('\n')
