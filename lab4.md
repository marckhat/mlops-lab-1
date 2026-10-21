# Lab 4 - Orchestrating the App with Docker Compose

> **Note on `Staging`:** the lab handout refers to moving the model to the `Staging` stage. Stages are deprecated in current MLflow (see Lab 3, Question 2), so this project keeps using the **`champion` alias**. Everywhere the handout says "move to Staging", I moved the `champion` alias instead, and `serve.py` keeps loading `models:/food11@champion`.

## Setup notes

Two changes were needed compared to the handout to make the stack work with MLflow 3.16:

1. **`--artifacts-destination` instead of `--default-artifact-root`** in `mlflow/Dockerfile`. With `--default-artifact-root /mlflow-data/mlruns`, the server only *tells* clients where to write artifacts, and each client writes to `/mlflow-data/mlruns` on its *own* filesystem (my Windows host, or the inference container), not on the volume. With `--artifacts-destination`, the server proxies artifacts over HTTP (`mlflow-artifacts:/` URIs), so every upload and download goes through the mlflow container and lands on the `mlflow-data` volume. This is the same problem I hit in Lab 3, where I had to register version 2 through an artifact-serving server.
2. **`--allowed-hosts`**. MLflow 3.x rejects requests whose `Host` header is not localhost or a private IP (protection against DNS-rebinding attacks). Requests from the inference container have `Host: mlflow:5000`, so `mlflow` and `mlflow:5000` were added to the allowed hosts, together with `localhost` and `127.0.0.1` for the browser.

The new mlflow container starts with an **empty** volume, so the models registered in Labs 2 and 3 (stored in the host's `mlflow.db` and `mlruns/`) are not in it. Instead of retraining, `scripts/register_model.py` loads an already-trained Lab 2 model from the local `mlruns/` folder, logs it to the containerized server, registers it as a new version of `food11` and moves the `champion` alias to it.

The readiness problem from Question 6 was handled with a **healthcheck** on the mlflow service (`depends_on: condition: service_healthy`) and `restart: on-failure` on the inference service.

---

## Question 1 - What happens to `/mlflow-data` without a volume?

Without a volume, everything written to `/mlflow-data` (the SQLite database `mlflow.db` and the `mlruns/` artifacts) is stored in the container's **writable layer**.

The writable layer belongs to that one container. It survives `docker stop` / `docker start` of the same container, but it is deleted when the container is removed (`docker rm`).

I tested it with the mlflow image built by Compose, on port 5001 so it would not clash with the running stack:

```powershell
docker run -d --name mlflow-test -p 5001:5000 mlops-lab-1-mlflow
# in the UI at http://127.0.0.1:5001 -> Experiments -> Create "no-volume-test"
docker rm -f mlflow-test
docker run -d --name mlflow-test -p 5001:5000 mlops-lab-1-mlflow
# refresh the UI
docker rm -f mlflow-test
```

* In the first container, the experiment list showed `no-volume-test` (created 04:14:13 PM) and `Default` (created 04:12:49 PM).
* In the new container started from the same image, only `Default` was left, and even `Default` had a new creation time (04:14:43 PM). The whole database had been created again from scratch.

The image only contains the MLflow installation, never the data. A new container starts with an empty `/mlflow-data`, so MLflow creates a new, empty `mlflow.db`. I also noticed that the first page load returned `ERR_EMPTY_RESPONSE` for a few seconds, because the server was still creating the new database before it started listening.

**In simple terms: without a volume, the data dies with the container.**

---

## Question 2 - Why a named volume instead of a bind mount?

A **named volume** (`mlflow-data`) is created and managed by Docker. It lives in Docker's own storage area, not in a specific folder on my computer.

Reasons to prefer it here:

* **Portability** - the compose file does not depend on a host path such as `C:\Users\User\...`, so it works the same on Windows, Mac, Linux and a server.
* **Reliability and speed** - on Docker Desktop for Windows/Mac, bind mounts cross the boundary between the host and Docker's Linux VM. That is slower and can cause file-locking problems with SQLite. A named volume lives inside the Linux VM.
* **Permissions** - Docker sets up the volume with the right ownership for the container; bind mounts can have host-permission issues.
* **Separation** - it keeps the server's database away from the repo. My repo already contains the Lab 2 `mlflow.db` and `mlruns/`, and a bind mount there could mix or overwrite them, or tempt someone to commit them.

A bind mount **would work** for local development, and it has one advantage: the files are easy to see and back up from the host. But it ties the setup to one machine's folder layout.

**In simple terms: a named volume is storage owned by Docker; a bind mount is a window into a folder on my computer.**

---

## Question 3 - Why does `http://mlflow:5000` resolve now?

Docker Compose creates a private network for the project (`mlops-lab-1_default`) and attaches every service to it.

Compose's network has an **embedded DNS server**. Each service is registered under its service name, so inside the network `mlflow` resolves to the IP address of the mlflow container.

In Lab 3, the inference container ran alone on Docker's default bridge network, and MLflow ran directly on Windows, not in a container. There was no container called `mlflow` and no service-name DNS, so the only way to reach the host was `host.docker.internal` or `--network host`.

**In simple terms: inside a Compose network, the service name is the hostname.**

---

## Question 4 - Why read `INFERENCE_URL` from an environment variable?

`http://inference:8000` only works **inside the Compose network**, where the service name `inference` exists.

If the frontend image is run on its own (`docker run -p 8501:8501 food11-frontend`), or the app is run locally with `streamlit run app.py`, the name `inference` cannot be resolved and every request would fail.

Reading the URL from an environment variable means the **same image** works everywhere. Only the configuration changes:

* Docker Compose: `INFERENCE_URL=http://inference:8000`
* Local development: the default `http://127.0.0.1:8000`
* Standalone container: e.g. `-e INFERENCE_URL=http://host.docker.internal:8000`
* Later in Kubernetes: the Kubernetes service name

This is the same idea as `MLFLOW_TRACKING_URI` in `serve.py`.

**In simple terms: build once, configure per environment.**

---

## Question 5 - Why does `inference` not publish a port?

A published port (`ports:`) is only needed when something **outside** Docker, such as my browser, must reach the service.

* `mlflow` publishes `5000` so I can open the MLflow UI and so `scripts/register_model.py`, running on the host, can register models.
* `frontend` publishes `8501` so I can open the Streamlit page.
* `inference` is only called by the frontend, never directly by a human.

The frontend reaches it through the Compose network at `http://inference:8000`. Service-to-service traffic goes directly to the container's port 8000 on the internal network, so no host port mapping is needed.

Not publishing it also reduces the attack surface: the API cannot be called directly from outside the machine.

**In simple terms: publish only what humans need; containers talk to each other on the internal network.**

---

## Question 6 - What happens if mlflow is not ready when `inference` starts?

`serve.py` loads `models:/food11@champion` at import time, before the API starts. If the mlflow container has started but the server is not yet accepting connections (or the registry is still empty), `mlflow.pyfunc.load_model` raises an exception.

The exception stops the uvicorn process, so the **inference container exits** with an error. If MLflow is not accepting connections yet, `docker compose logs inference` shows a connection error (`Connection refused` / `Max retries exceeded ... mlflow:5000`). With only `depends_on: [mlflow]`, nothing restarts the container, and the frontend gets errors when it calls the API.

I saw the same failure in practice on the very first `docker compose up`, before any model had been registered in the new MLflow container. `docker compose ps` showed `inference` as `Restarting (1)`, and the logs showed:

```
MLflow tracking URI: http://mlflow:5000
Loading model: models:/food11@champion
mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: Registered Model with name=food11 not found
```

As soon as I registered a model, the next automatic restart succeeded and `inference` stayed `Up`.

To fix this in my compose file:

* The mlflow service has a **healthcheck** that calls `http://localhost:5000/health`.
* The inference service uses `depends_on: mlflow: condition: service_healthy`, so it waits until MLflow actually answers.
* The inference service has `restart: on-failure`, so if the model still cannot be loaded (for example the registry is still empty on the very first start), Docker restarts it until `food11@champion` exists.

The `docker compose up` output shows the healthcheck working: `mlflow-1 Healthy 30.0s`, and only then `inference-1 Started 30.3s`.

**In simple terms: `depends_on` waits for the container to start; a healthcheck waits for the service to be ready.**

---

## Question 7 - What does `docker compose ps` show?

```
NAME                      SERVICE     STATUS                    PORTS
mlops-lab-1-frontend-1    frontend    Up 13 minutes             0.0.0.0:8501->8501/tcp, [::]:8501->8501/tcp
mlops-lab-1-inference-1   inference   Up 3 minutes              8000/tcp
mlops-lab-1-mlflow-1      mlflow      Up 14 minutes (healthy)   0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp
```

* `mlflow` -> `0.0.0.0:5000->5000/tcp`: published, and `healthy` thanks to the healthcheck
* `frontend` -> `0.0.0.0:8501->8501/tcp`: published
* `inference` -> only `8000/tcp`, with no `0.0.0.0:...->` mapping

`8000/tcp` appears because the inference Dockerfile has `EXPOSE 8000`, but `EXPOSE` is only documentation and does not make the port reachable from the host.

This matches the `docker-compose.yml`: only `mlflow` and `frontend` have a `ports:` section.

---

## Question 8 - Does the frontend use the new model version?

I first registered the runner-up model (`invincible-shad-365`, lr 0.001, batch size 64, 61.2% validation accuracy) as version 1 with the `champion` alias. I then registered the best model (`skillful-midge-64`, lr 0.0001, batch size 32, 72.6% validation accuracy) as version 2 and moved `champion` to it.

```bash
uv run python scripts/register_model.py --model best
```

I used the same test image every time: `data/food11_processed_mini/validation/Bread/0_0.jpg`.

| Moment | Champion in registry | Model used by inference | Frontend prediction |
|---|---|---|---|
| After registering version 1 | Version 1 | Version 1 | Fried food (99.0%) |
| After moving `champion` to version 2, **no restart** | Version 2 | Version 1 | Fried food (99.0%) |
| After `docker compose restart inference` | Version 2 | Version 2 | Fried food (61.3%) |

After moving the alias and uploading the same image again, the prediction still came from **the old model (version 1)**: exactly the same output, 99.0%. The registry page already showed `@champion` on version 2.

`serve.py` resolves `food11@champion` and loads the model **once, at startup**. Moving the alias in the registry does not affect a model that is already in memory.

The single command to pick up the new version is:

```bash
docker compose restart inference
```

After the restart, the same image gave **61.3%** instead of 99.0%, so version 2 was now being used. This also matches Lab 3: the same `skillful-midge-64` model returned `Fried food` with confidence `0.6134` for this image.

Both models predict the wrong class here (the image is in the `Bread` folder). The better model is much less confident in its mistake, but the important point for this question is that the restart changed which model was serving, without rebuilding any image.

---

## Question 9 - Why does `restart` work without a rebuild?

The inference **image** contains only the Python environment, the dependencies, the `src/` code and the startup command. The **model is not baked into the image**. It is downloaded from MLflow every time the container starts, by resolving `models:/food11@champion`.

Restarting the container re-runs `serve.py`, which asks MLflow which version `champion` points to *now* and downloads that version. Nothing in the image changed, so there is nothing to rebuild.

This confirms the conclusion of Lab 3, Question 8: the application is baked into the image, while the model is configuration and data fetched at runtime. Model updates (moving an alias) and code updates (rebuilding the image) are separate, independent workflows.

---

## Question 10 - What survives `down`/`up` and `down -v`?

**`docker compose down` then `docker compose up`:**

Yes. The output of `down` only listed the three containers and the network as `Removed`, with no volume. After `up`, the MLflow registry still showed `food11` with Version 1, Version 2 and `@champion` on Version 2, and the frontend still predicted **Fried food (61.3%)**.

`down` removes the containers and the network, but **not named volumes**. The new mlflow container mounted the same `mlflow-data` volume containing `mlflow.db` and the artifacts, and the inference service loaded the champion model again.

**`docker compose down -v` then `docker compose up`:**

`-v` also deletes the named volumes declared in the compose file. This time the output included `Volume mlops-lab-1_mlflow-data Removed`, and `up` showed `Volume mlops-lab-1_mlflow-data Created`: a brand-new, empty volume. The database and all model artifacts were gone.

* **MLflow:** opening the `food11` model page showed `404 - Model food11 does not exist`.
* **Inference:** `docker compose ps` showed `Restarting (1)`, and the logs showed `RESOURCE_DOES_NOT_EXIST: Registered Model with name=food11 not found`.
* **Frontend:** `Could not reach the inference service at http://inference:8000 ... NameResolutionError ... Failed to resolve 'inference'`. Because the inference container keeps crashing and restarting, it is not running long enough to be registered in Compose's DNS, so even the name `inference` cannot be resolved.

I had to run `scripts/register_model.py --model best` again to restore a champion model.

**In simple terms: `down` throws away the containers; `down -v` also throws away the data.**

---

## Question 11 - What can Docker Compose not provide?

Docker Compose runs all services on **one machine** with **one Docker engine**. It cannot provide:

* **Multi-machine scheduling** - spreading containers across several nodes and moving them to a healthy node if a machine fails.
* **Replicas with built-in load balancing and service discovery** - Compose can start several copies with `--scale inference=3`, but only on the same machine, without a production-grade load balancer or rolling updates.
* **Self-healing at cluster level** - `restart:` only restarts a container on the same host. If the host dies, everything dies.
* **Rolling updates and rollbacks** - replacing replicas one at a time without downtime.
* **Autoscaling** - adding replicas automatically based on CPU usage or request load.
* **High availability for MLflow** - surviving a machine failure needs storage that is not tied to one host: a managed or replicated database (e.g. PostgreSQL instead of SQLite on a local volume), artifacts in shared object storage (e.g. S3/MinIO/Azure Blob), and several stateless MLflow server replicas behind a load balancer.

These are the problems a container orchestrator such as **Kubernetes** solves, with Deployments (replicas, rolling updates), Services (load balancing, DNS), probes (readiness and liveness), and PersistentVolumes or external storage.

**In simple terms: Compose orchestrates containers on one machine; Kubernetes orchestrates them across a cluster.**
