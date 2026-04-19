web: find /workspace -name '_cffi__*.c' -delete 2>/dev/null; find /workspace -name '_cffi__*.so' -delete 2>/dev/null; find . -name '_cffi__*.c' -delete 2>/dev/null; find . -name '_cffi__*.so' -delete 2>/dev/null; gunicorn wsgi_config:app --bind 0.0.0.0:${PORT:-8000} --config gunicorn_conf.py
sse: gunicorn sse_server:app --bind 0.0.0.0:${SSE_PORT:-8001} --config sse_gunicorn_conf.py
