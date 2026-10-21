FROM python:3.11-slim

# Pin the same mlflow version as the project (uv.lock) so the server and
# the clients (training script, inference API) speak the same version.
RUN pip install --no-cache-dir mlflow==3.16.0

EXPOSE 5000

# --artifacts-destination (instead of --default-artifact-root) makes the server
#   proxy artifacts over HTTP (mlflow-artifacts:/ URIs). Clients on the host and
#   in other containers upload/download models through the server, and the files
#   land on the /mlflow-data volume. With a plain local path as artifact root,
#   each client would try to write to /mlflow-data on its OWN filesystem.
# --allowed-hosts: MLflow 3.x only accepts localhost/private-IP Host headers by
#   default, so requests to http://mlflow:5000 from other containers must be allowed.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--artifacts-destination", "/mlflow-data/mlruns", \
     "--allowed-hosts", "mlflow,mlflow:5000,localhost,localhost:*,127.0.0.1,127.0.0.1:*"]
