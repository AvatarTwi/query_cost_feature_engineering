def name_align(name):
    name = name.replace("snapshot_model_filter/save_model_QPPNet", "FSO(QPPNet)")
    name = name.replace("snapshot_model_FR/save_model_QPPNet", "FSO(QPPNet)")
    name = name.replace("snapshot_model_filter/save_model_MSCN", "FSO(MSCN)")
    name = name.replace("snapshot_model_FR/save_model_MSCN", "FSO(MSCN)")
    name = name.replace("origin_model/save_model_QPPNet", "QPPNet")
    name = name.replace("origin_model/save_model_MSCN", "MSCN")
    return name

def sort(plan, dict_alls):
    for k in dict_alls.keys():
        dict_all = dict_alls[k]

        dict_all_new = {}
        for idy, key in enumerate(dict_all.keys()):
            dict_all_new[plan['name_align'](key)] = dict_all[key]
        dict_alls[k] = dict(sorted(dict_all_new.items(), key=lambda x: x[0]))
