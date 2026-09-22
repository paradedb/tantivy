\set ON_ERROR_STOP on
CREATE OR REPLACE FUNCTION public.diagnostic_inline_posting_norms(regclass,text) RETURNS jsonb
AS '$libdir/pg_search', 'diagnostic_inline_posting_norms_wrapper' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
REVOKE ALL ON FUNCTION public.diagnostic_inline_posting_norms(regclass,text) FROM PUBLIC;
