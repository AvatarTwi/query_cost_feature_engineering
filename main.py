import os

import config
from config import *
from build_dataset import build_ds
from build_model import build_md
from utils.opt_parser import getParser, defaultParser
from utils.util import Utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dstype_type_dict = {
    'TPCH': 'PSQLTPCH',
    'Sysbench': 'PSQLSysbench',
    'job-light': 'PSQLJOB',
}

data_dir_dict = {
    'TPCH': './data_dir/tpch',
    'Sysbench': './data_dir/sysbench',
    'job-light': './data_dir/job',
    'TPCH_transfer': './data_dir/tpch_transfer',
    'job-light_transfer': './data_dir/job_transfer',
}

start_epoch = {
    'TPCH': 0,
    'Sysbench': 0,
    'job-light': 0,
}

end_epoch = {
    'TPCH': 400,
    'Sysbench': 100,
    'job-light': 800,
}

if __name__ == '__main__':

    opt = defaultParser().parse_args()
    benchmark_type = opt.workload
    if benchmark_type == 'qcfe':
        benchmark_type = 'knob_model_shap'
    model_type = opt.model
    model = opt.type
    version = str(opt.scale) + "/" + benchmark_type

    Utils.path_build("./" + version)
    new_md = True if model_type == 'QPPNet' or model_type == 'MSCN' else False

    if 'knob_model_' in model:

        filter_type = model.replace("knob_model_", "")

        for i in range(2):
            if os.path.exists("./2000/" + benchmark_type + '/knob_model/save_model_'
                              + model_type + "/1024/" + filter_type + "_values_array.pickle"):
                version_sp = version
            else:
                version_sp = "2000/" + benchmark_type

            if not os.path.exists("./2000/" + benchmark_type + '/knob_model/save_model_' + model_type):
                opt = getParser(version=version_sp,
                                dataset=dstype_type_dict[benchmark_type],
                                new_ds=False, new_md=False,
                                mid_data_dir='./' + version_sp + '/knob_model',
                                data_structure='./' + version_sp + '/data_structure',
                                data_dir=data_dir_dict[benchmark_type],
                                saved_model='/save_model_' + model_type,
                                mode='train',
                                start_epoch=start_epoch[benchmark_type],
                                end_epoch=end_epoch[benchmark_type],
                                scale=opt.scale).parse_args()
                dataset, dim_dict = build_ds(opt, 'knob_model')
                build_md(dataset, model_type, opt, dim_dict)

            opt = getParser(version=version_sp,
                            dataset=dstype_type_dict[benchmark_type],
                            new_ds=False, new_md=False,
                            mid_data_dir='./' + version_sp + '/knob_model',
                            data_structure='./' + version_sp + '/data_structure',
                            data_dir=data_dir_dict[benchmark_type],
                            saved_model='/save_model_' + model_type,
                            mode=filter_type + '_eval',
                            scale=opt.scale).parse_args()

            if os.path.exists("./2000/" + benchmark_type + '/knob_model/save_model_'
                              + model_type + "/1024/" + filter_type + "_values_array.pickle"):
                dataset, dim_dict = build_ds(opt, 'knob_model')
                break
            else:
                dataset, dim_dict = build_ds(opt, 'knob_model')
                build_md(dataset, model_type, opt, dim_dict)

        opt = getParser(version=version,
                        dataset=dstype_type_dict[benchmark_type],
                        new_ds=False, new_md=False,
                        mid_data_dir='./' + version + '/' + model,
                        data_structure='./' + version + '/data_structure',
                        data_dir=data_dir_dict[benchmark_type],
                        saved_model='/save_model_' + model_type,
                        mode='train',
                        start_epoch=start_epoch[benchmark_type],
                        end_epoch=end_epoch[benchmark_type],
                        scale=opt.scale).parse_args()

        Utils.path_build(opt.mid_data_dir)
        Utils.path_build(opt.data_structure)

        build_md(dataset, model_type, opt, dim_dict)

    elif "origin_model" in model:
        opt = getParser(version=version,
                        dataset=dstype_type_dict[benchmark_type],
                        new_ds=True if model_type == 'QPPNet' else False,
                        new_md=new_md,
                        mid_data_dir='./' + version + '/' + model,
                        data_structure='./' + version + '/data_structure' + model.replace(
                            "knob_model_template", "").replace("origin_model", ""),
                        data_dir=data_dir_dict[benchmark_type],
                        saved_model='/save_model_' + model_type,
                        mode='train',
                        start_epoch=start_epoch[benchmark_type],
                        end_epoch=end_epoch[benchmark_type],
                        scale=opt.scale).parse_args()

        dataset, dim_dict = build_ds(opt, model)
        build_md(dataset, model_type, opt, dim_dict)

    elif 'transfer' not in model:
        opt = getParser(version=version,
                        dataset=dstype_type_dict[benchmark_type],
                        new_ds=True if 'template' in model else False,
                        new_md=new_md,
                        mid_data_dir='./' + version + '/' + model,
                        data_structure='./' + version + '/data_structure' + model.replace(
                            "knob_model", "").replace("origin_model", ""),
                        data_dir=data_dir_dict[benchmark_type],
                        saved_model='/save_model_' + model_type,
                        mode='train',
                        start_epoch=start_epoch[benchmark_type],
                        end_epoch=end_epoch[benchmark_type],
                        scale=opt.scale).parse_args()

        dataset, dim_dict = build_ds(opt, model)
        build_md(dataset, model_type, opt, dim_dict)

    else:
        opt = getParser(version=version,
                        dataset=dstype_type_dict[benchmark_type],
                        new_ds=False, new_md=False,
                        mid_data_dir='./' + version + '/' + model,
                        data_structure='./' + version + '/data_structure_' + model
                        .replace("knob_model", ""),
                        data_dir=data_dir_dict[benchmark_type + "_transfer"],
                        saved_model='/save_model_' + model_type,
                        mode='change_train',
                        start_epoch=start_epoch[benchmark_type],
                        end_epoch=end_epoch[benchmark_type],
                        scale=opt.scale).parse_args()

        dataset, dim_dict = build_ds(opt, model)
        Utils.path_build(opt.mid_data_dir)
        build_md(dataset, model_type, opt, dim_dict)
