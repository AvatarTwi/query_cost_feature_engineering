num_per_q = [1, 2, 3, 4]

template_num = 1000

total = 0.0


def set_per_qs(new):
    global num_per_q
    num_per_q = new


cost_factor_dict = {'Seq Scan': 2,
                    'Index Scan': 2,
                    'Index Only Scan': 2,
                    'Bitmap Heap Scan': 2,
                    'Bitmap Index Scan': 2,
                    'Sort': 2,
                    'Hash': 2,
                    'Hash Join': 2,
                    'Merge Join': 2,
                    'Aggregate': 2,
                    'Nested Loop': 4,
                    'Limit': 1,
                    'Subquery Scan': 1,
                    'Materialize': 2,
                    'Gather Merge': 2,
                    'Gather': 2,
                    'BitmapAnd': 1,
                    'Memoize': 2,
                    'ModifyTable': 2,
                    'Result': 2,
                    'LockRows': 2,
                    'Append': 2,
                    'Unique': 2,
                    }

filter_type = ['shap', 'grad', 'R2']
