FROM tf_2_14:latest

WORKDIR /app
COPY server.py /app/server.py

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000", "--timeout-keep-alive", "600"]
