\set ON_ERROR_STOP on
CREATE FUNCTION public.diagnostic_subblock_pruning(boolean)
RETURNS boolean
AS '$libdir/pg_search', 'diagnostic_subblock_pruning_wrapper'
LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
