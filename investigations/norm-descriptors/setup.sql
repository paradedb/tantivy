\set ON_ERROR_STOP on
CREATE OR REPLACE FUNCTION public.diagnostic_embedded_norm_directory(boolean) RETURNS boolean
AS '$libdir/pg_search', 'diagnostic_embedded_norm_directory_wrapper' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
CREATE OR REPLACE FUNCTION public.diagnostic_rewrite_norm_directory(regclass, text) RETURNS jsonb
AS '$libdir/pg_search', 'diagnostic_rewrite_norm_directory_wrapper' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
CREATE OR REPLACE FUNCTION public.diagnostic_select_norm_components(regclass, jsonb) RETURNS jsonb
AS '$libdir/pg_search', 'diagnostic_select_norm_components_wrapper' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;

REVOKE ALL ON FUNCTION public.diagnostic_rewrite_norm_directory(regclass,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.diagnostic_select_norm_components(regclass,jsonb) FROM PUBLIC;
