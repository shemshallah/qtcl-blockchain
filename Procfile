web: gunicorn -w 2 --worker-class gthread --threads 4 --timeout 120 --graceful-timeout 30 --keep-alive 5 --max-requests 1000 --max-requests-jitter 100 --access-logfile - --error-logfile - -b 0.0.0.0:${PORT:-5000} wsgi_config:application
release: python db_builder_v2.py
