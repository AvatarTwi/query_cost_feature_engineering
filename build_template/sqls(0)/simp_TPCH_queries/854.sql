EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON)  SELECT *  FROM lineitem  WHERE   lineitem.l_extendedprice >= 32440  ORDER BY l_partkey 