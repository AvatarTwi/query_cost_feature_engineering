EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM partsupp  JOIN part  ON partsupp.ps_partkey = part.p_partkey  WHERE   partsupp.ps_suppkey >= 605 