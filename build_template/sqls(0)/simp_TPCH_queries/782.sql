EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM lineitem  WHERE   lineitem.l_partkey >= 1493  ORDER BY l_linestatus 