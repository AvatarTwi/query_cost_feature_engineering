# all operators used in tpc-h
all_dicts = ['Aggregate', 'Gather Merge', 'Sort', 'Seq Scan', 'Index Scan',
             'Index Only Scan', 'Bitmap Heap Scan', 'Bitmap Index Scan',
             'Limit', 'Hash Join', 'Hash', 'Nested Loop', 'Materialize',
             'Merge Join', 'Subquery Scan', 'Gather', 'BitmapAnd', 'Memoize'
            ,'ModifyTable', 'LockRows', 'Result', 'Append', 'Unique']

operator_list = \
[
                'Seq Scan',
                'Index Scan',
                'Index Only Scan',
                'Bitmap Heap Scan',
                'Bitmap Index Scan',
                'Sort',
                'Hash',
                'Hash Join', 'Merge Join',
                'Aggregate', 'Nested Loop', 'Limit',
                'Subquery Scan', 'BitmapAnd',
                'Materialize', 'Gather Merge', 'Gather',
                'Memoize'
]
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

