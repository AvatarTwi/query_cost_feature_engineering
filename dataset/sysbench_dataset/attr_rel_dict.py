# all operators used in tpc-h
all_dicts = ['Aggregate', 'Gather Merge', 'Sort', 'Seq Scan', 'Index Scan',
             'Index Only Scan', 'Bitmap Heap Scan', 'Bitmap Index Scan',
             'Limit', 'Hash Join', 'Hash', 'Nested Loop', 'Materialize',
             'Merge Join', 'Subquery Scan', 'Gather', 'BitmapAnd', 'Memoize'
             ,'ModifyTable','LockRows','Result','Append','Unique']

join_types = ['semi', 'inner', 'anti', 'full', 'right', 'left']

parent_rel_types = ['inner', 'outer', 'subquery']

sort_algos = ['quicksort', 'top-n heapsort']

aggreg_strats = ['plain', 'sorted', 'hashed']

rel_names = ['sbtest1']

index_names = ['sbtest1_pkey']

rel_attr_list_dict = \
    {
        'sbtest1':[
            "id",
            "k",
            "c",
            "pad"
        ]
    }
