\set ON_ERROR_STOP on
CREATE FUNCTION public.diagnostic_packed_posting_norms(boolean) RETURNS boolean
AS '$libdir/pg_search', 'diagnostic_packed_posting_norms_wrapper' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
CREATE FUNCTION public.diagnostic_pack_posting_norms(regclass) RETURNS jsonb
AS '$libdir/pg_search', 'diagnostic_pack_posting_norms_wrapper' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
