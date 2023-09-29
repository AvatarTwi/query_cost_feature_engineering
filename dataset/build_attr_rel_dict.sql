
select
	'"'||a.table_name||'":[', array_to_string(array_agg('"'||a.column_name||'"'),',')||'],'
from
	information_schema.columns as a
where
	table_schema = 'public'
GROUP BY a.table_name
ORDER BY table_name;

select
	table_name, ',')'"'||column_name||'"'||','
from
	information_schema.columns
where
	table_schema = 'public'
ORDER BY "table_name";

select '"'||b.table_name||'"'||','
FROM
(
select
	DISTINCT a.table_name
from
	information_schema.columns a
where
	table_schema = 'public'
ORDER BY "table_name") as b;

select
a.schemaname,
a.tablename,
'"'||a.indexname||'"'||','
from pg_indexes a
WHERE a.schemaname = 'public'
order by a.tablename

