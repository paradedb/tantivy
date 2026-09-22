\set ON_ERROR_STOP on
CREATE FUNCTION public.diagnostic_posting_norms(boolean) RETURNS boolean
AS '$libdir/pg_search', 'diagnostic_posting_norms_wrapper' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
CREATE FUNCTION public.diagnostic_posting_norm_reads() RETURNS bigint
AS '$libdir/pg_search', 'diagnostic_posting_norm_reads_wrapper' LANGUAGE C VOLATILE PARALLEL UNSAFE;
CREATE FUNCTION public.diagnostic_subblock_pruning(boolean) RETURNS boolean
AS '$libdir/pg_search', 'diagnostic_subblock_pruning_wrapper' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
CREATE FUNCTION public.diagnostic_posting_norm_storage(regclass) RETURNS jsonb
AS '$libdir/pg_search', 'diagnostic_posting_norm_storage_wrapper' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
