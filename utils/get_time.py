import os

from dataset.postgres_tpch_dataset.cost_factor.cost_factor import get_all_plans as tpch_get
from dataset.job_dataset.cost_factor.cost_factor import get_all_plans as job_get

from dataset.sysbench_dataset.cost_factor.cost_factor import get_all_plans as sys_get

ds_dict = {
    # 'tpch': {
    #     'dir': "../res_by_dir/tpch",
    #     "num_q": 22,
    #     "get_all_plan": tpch_get
    # },
    'tpch_template': {
        'dir': "../res_by_dir/tpch_template",
        "num_q": 1,
        "get_all_plan": tpch_get
    },
    # 'tpch_i7': {
    #     'dir': "../res_by_dir/tpch_i7",
    #     "num_q": 22,
    #     "get_all_plan": tpch_get
    # },
    # 'tpch_i7_template': {
    #     'dir': "../res_by_dir/tpch_i7_template",
    #     "num_q": 1,
    #     "get_all_plan": tpch_get
    # },
    # 'sysbench': {
    #     'dir': "../res_by_dir/sysbench",
    #     "num_q": 1,
    #     "get_all_plan": sys_get
    # },
    # 'sysbench_template': {
    #     'dir': "../res_by_dir/sysbench_template",
    #     "num_q": 1,
    #     "get_all_plan": sys_get
    # },
    # 'sysbench_i7': {
    #     'dir': "../res_by_dir/sysbench_i7",
    #     "num_q": 1,
    #     "get_all_plan": sys_get
    # },
    # 'sysbench_i7_template': {
    #     'dir': "../res_by_dir/sysbench_i7_template",
    #     "num_q": 1,
    #     "get_all_plan": sys_get
    # },
    # 'job': {
    #     'dir': "../res_by_dir/job",
    #     "num_q": 1,
    #     "get_all_plan": job_get
    # },
    # 'job_template': {
    #     'dir': "../res_by_dir/job_template",
    #     "num_q": 1,
    #     "get_all_plan": tpch_get
    # },
    # 'job_i7': {
    #     'dir': "../res_by_dir/job_i7",
    #     "num_q": 1,
    #     "get_all_plan": job_get
    # },
    # 'job_i7_template': {
    #     'dir': "../res_by_dir/job_i7_template",
    #     "num_q": 1,
    #     "get_all_plan": tpch_get
    # },
}

run_time = {
    'tpch': 27524503.449999925,
    'sysbench': 50211.28399991205,
    'job': 114490268.76199947,
    'tpch_template': 34251684.562999964,
    'sysbench_template': 47806.017999913354,
    'job_template': 15628359.509000026,
    "tpch_i7":10816345.969999962,
    "tpch_i7_template":5885229.50699999,
    "sysbench_i7":45275.49399989756,
    "sysbench_i7_template":45275.49399989756,
    "job_i7":11330183.754000004,
    "job_i7_template":1004845.2839999995,
}

if __name__ == '__main__':
    for key in ds_dict.keys():
        data_dir = ds_dict[key]['dir']
        num_q = ds_dict[key]['num_q']
        total_time = 0
        for root, dirs, files in os.walk(data_dir):
            for dir in dirs:
                temp_data = []
                for i in range(num_q):
                    try:
                        fname = root + "/" + dir + "/q" + str(i + 1) + ".json"
                        data1 = ds_dict[key]["get_all_plan"](fname)
                    except:
                        fname = root + "/" + dir + "/serverlog"
                        data1 = ds_dict[key]["get_all_plan"](fname)
                    for data in data1:
                        total_time += data['Actual Total Time']
        print(key, total_time)

