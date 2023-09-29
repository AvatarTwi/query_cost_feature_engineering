# all operators used in tpc-h
import os
import random
from time import sleep

all_dicts = ['Aggregate', 'Gather Merge', 'Sort', 'Seq Scan', 'Index Scan',
             'Index Only Scan', 'Bitmap Heap Scan', 'Bitmap Index Scan',
             'Limit', 'Hash Join', 'Hash', 'Nested Loop', 'Materialize',
             'Merge Join', 'Subquery Scan', 'Gather', 'BitmapAnd', 'Memoize'
    , 'ModifyTable', 'LockRows', 'Result', 'Append', 'Unique']

join_types = ['semi', 'inner', 'anti', 'full', 'right', 'left']

parent_rel_types = ['inner', 'outer', 'subquery']

sort_algos = ['quicksort', 'top-n heapsort']

aggreg_strats = ['plain', 'sorted', 'hashed']

rel_names = ["aka_name",
             "aka_title",
             "cast_info",
             "char_name",
             "comp_cast_type",
             "company_name",
             "company_type",
             "complete_cast",
             "info_type",
             "keyword",
             "kind_type",
             "link_type",
             "movie_companies",
             "movie_info",
             "movie_info_idx",
             "movie_keyword",
             "movie_link",
             "name",
             "person_info",
             "role_type",
             "title", ]

index_names = ["aka_name_pkey",
               "aka_title_pkey",
               "cast_info_pkey",
               "char_name_pkey",
               "comp_cast_type_pkey",
               "company_name_pkey",
               "company_type_pkey",
               "complete_cast_pkey",
               "info_type_pkey",
               "keyword_pkey",
               "kind_type_pkey",
               "link_type_pkey",
               "movie_companies_pkey",
               "movie_info_pkey",
               "movie_info_idx_pkey",
               "movie_keyword_pkey",
               "movie_link_pkey",
               "name_pkey",
               "person_info_pkey",
               "role_type_pkey",
               "title_pkey", ]

rel_attr_list_dict = \
    {
        "aka_name": ["id", "person_id", "name", "imdb_index", "name_pcode_cf", "name_pcode_nf", "surname_pcode",
                     "md5sum"],
        "aka_title": ["id", "movie_id", "title", "imdb_index", "kind_id", "production_year", "phonetic_code",
                      "episode_of_id", "season_nr", "episode_nr", "note", "md5sum"],
        "cast_info": ["id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"],
        "char_name": ["id", "name", "imdb_index", "imdb_id", "name_pcode_nf", "surname_pcode", "md5sum"],
        "comp_cast_type": ["id", "kind"],
        "company_name": ["id", "name", "country_code", "imdb_id", "name_pcode_nf", "name_pcode_sf", "md5sum"],
        "company_type": ["id", "kind"],
        "complete_cast": ["id", "movie_id", "subject_id", "status_id"],
        "info_type": ["id", "info"],
        "keyword": ["id", "keyword", "phonetic_code"],
        "kind_type": ["id", "kind"],
        "link_type": ["id", "link"],
        "movie_companies": ["company_type_id", "note", "id", "movie_id", "company_id"],
        "movie_info": ["movie_id", "id", "note", "info", "info_type_id"],
        "movie_info_idx": ["info_type_id", "id", "movie_id", "note", "info"],
        "movie_keyword": ["id", "movie_id", "keyword_id"],
        "movie_link": ["link_type_id", "linked_movie_id", "movie_id", "id"],
        "name": ["name", "id", "imdb_index", "imdb_id", "gender", "name_pcode_cf", "name_pcode_nf", "surname_pcode",
                 "md5sum"],
        "person_info": ["person_id", "info", "note", "id", "info_type_id"],
        "role_type": ["id", "role"],
        "title": ["series_years", "md5sum", "kind_id", "title", "id", "imdb_index", "production_year", "imdb_id",
                  "phonetic_code", "episode_of_id", "season_nr", "episode_nr"],
    }

attr_val_dict = {'min': {'aka_name': [1.0, 4.0, 0, 0, 0, 0, 0, 0],
                         'aka_title': [1.0, 0.0, 0, 0, 1.0, 1888.0, 0, 10.0, 1.0, 1.0, 0, 0],
                         'cast_info': [1.0, 1.0, 1.0, 1.0, 0, 0.0, 1.0], 'char_name': [1.0, 0, 0, 0, 0, 0],
                         'comp_cast_type': [1.0, 0], 'company_name': [1.0, 0, 0, 0, 0, 0], 'company_type': [1.0, 0],
                         'complete_cast': [1.0, 285.0, 1.0, 3.0], 'info_type': [1.0, 0], 'keyword': [1.0, 0, 0],
                         'kind_type': [1.0, 0], 'link_type': [1.0, 0], 'movie_companies': [1.0, 0, 1.0, 2.0, 1.0],
                         'movie_info': [1.0, 1.0, 0, 0, 1.0], 'movie_info_idx': [99.0, 1.0, 2.0, 0, 0],
                         'movie_keyword': [1.0, 2.0, 1.0], 'movie_link': [1.0, 284.0, 2.0, 1.0],
                         'name': [0, 1.0, 0, 0, 0, 0, 0, 0], 'person_info': [4.0, 0, 0, 1.0, 15.0],
                         'role_type': [1.0, 0], 'title': [0, 0, 1.0, 0, 1.0, 0, 1880.0, 0, 0, 2.0, 1.0, 1.0]},
                 'max': {'aka_name': [901343.0, 4167489.0, 0, 0, 0, 0, 0, 0],
                         'aka_title': [377960.0, 2525672.0, 0, 0, 7.0, 2016.0, 0, 25391.0, 63.0, 4315.0, 0, 0],
                         'cast_info': [36244344.0, 4061926.0, 2525975.0, 3140339.0, 0, 1115798165.0, 11.0],
                         'char_name': [3140339.0, 0, 0, 0, 0, 0], 'comp_cast_type': [4.0, 0],
                         'company_name': [234997.0, 0, 0, 0, 0, 0], 'company_type': [4.0, 0],
                         'complete_cast': [135086.0, 2528312.0, 2.0, 4.0], 'info_type': [113.0, 0],
                         'keyword': [134170.0, 0, 0], 'kind_type': [7.0, 0], 'link_type': [18.0, 0],
                         'movie_companies': [2.0, 0, 2609129.0, 2525745.0, 234997.0],
                         'movie_info': [2526430.0, 14835720.0, 0, 0, 110.0],
                         'movie_info_idx': [113.0, 1380035.0, 2525793.0, 9997.0, 0],
                         'movie_keyword': [4523930.0, 2525971.0, 134170.0],
                         'movie_link': [17.0, 2524994.0, 186175.0, 29997.0], 'name': [0, 4167491.0, 0, 0, 0, 0, 0, 0],
                         'person_info': [4167491.0, 0, 0, 2963664.0, 39.0], 'role_type': [12.0, 0],
                         'title': [0, 0, 7.0, 0, 2528312.0, 0, 2019.0, 0, 0, 2528186.0, 2013.0, 91821.0]},
                 'med': {'aka_name': [450672.0, 2058068.0, 0, 0, 0, 0, 0, 0],
                         'aka_title': [190107.0, 2051198.0, 0, 0, 1.0, 1983.0, 0, 12159.0, 1.0, 8.0, 0, 0],
                         'cast_info': [18122172.0, 1798366.0, 1311458.0, 344978.0, 0, 7.0, 2.0],
                         'char_name': [1570170.0, 0, 0, 0, 0, 0], 'comp_cast_type': [2.0, 0],
                         'company_name': [117499.0, 0, 0, 0, 0, 0], 'company_type': [2.0, 0],
                         'complete_cast': [67543.0, 1697979.0, 1.0, 3.0], 'info_type': [57.0, 0],
                         'keyword': [67085.0, 0, 0], 'kind_type': [4.0, 0], 'link_type': [9.0, 0],
                         'movie_companies': [2.0, 0, 1304565.0, 1844005.0, 11318.0],
                         'movie_info': [1878132.0, 7417860.0, 0, 0, 8.0],
                         'movie_info_idx': [100.0, 690018.0, 1795802.0, 33.0, 0],
                         'movie_keyword': [2261965.0, 1993455.0, 4703.0],
                         'movie_link': [6.0, 1645852.0, 97052.0, 14999.0], 'name': [0, 2083746.0, 0, 0, 0, 0, 0, 0],
                         'person_info': [1591351.0, 0, 1481832.0, 22.0], 'role_type': [6.0, 0],
                         'title': [0, 0, 7.0, 0, 1264156.0, 0, 2003.0, 0, 816853.0, 1.0, 11.0]}}


class JOB:

    @staticmethod
    def random_table_attr_value(table=0, attr=0):
        while True:
            try:
                if table == 0:
                    table = random.choice(list(rel_attr_list_dict.keys()))
                attr = random.choice(rel_attr_list_dict[table])
                id = rel_attr_list_dict[table].index(attr)
                min = int(attr_val_dict['min'][table][id])
                max = int(attr_val_dict['max'][table][id])
                val = random.randint(min, max)
                if attr_val_dict['min'][table][id] != 0 or attr_val_dict['max'][table][id] != 0:
                    break
            except:
                continue

        return table, attr, val

    @staticmethod
    def random_get_join(table1=0):
        while True:
            if table1 == 0:
                table1 = random.choice(list(rel_attr_list_dict.keys()))

            for attr in rel_attr_list_dict[table1]:
                for table2 in rel_attr_list_dict.keys():
                    if table2 == table1:
                        continue
                    if attr not in rel_attr_list_dict[table2]:
                        continue
                    else:
                        break
                break
            if attr in rel_attr_list_dict[table2]:
                break
        attr1 = attr
        attr2 = attr
        return table1, table2, attr1, attr2

    @staticmethod
    def collect_table_attr(root='job_queries'):
        file_dirs = [os.path.join(root, file) for file in os.listdir(root)]
        dict = {}
        joinon = set()
        groupby = {}
        orderby = {}
        for file in file_dirs:
            with open(file, 'r+') as f:
                for line in f.readlines():
                    file_ = line.split("FROM")[-1]
                    join_on = file_.split("WHERE")[0]
                    for rel in rel_attr_list_dict.keys():
                        if rel in file_:
                            if rel not in dict.keys():
                                dict[rel] = set()
                        else:
                            continue
                        for attr in rel_attr_list_dict[rel]:
                            if attr in file_:
                                dict[rel].add(attr)
                    rels = []
                    for rel in rel_attr_list_dict.keys():
                        if " " + rel in join_on or "," + rel in join_on:
                            rels.append(rel)
                    for rel1 in rels:
                        if rel1 != 'title':
                            joinon.add(('title', rel1, 'id', 'movie_id'))

        return dict, joinon, groupby, orderby
