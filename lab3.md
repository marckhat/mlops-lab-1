# Lab 3 - Containerizing the Model with Docker

## Question 1 - What version was the model given, and what is the difference?

The best model was initially registered under the name `food11` and assigned version **1**.

The logged model artifact is the model produced by one specific training run. It remains connected to that run and its parameters and metrics.

A registered model gives the model a shared name and a separate version number. This makes it easier to manage and serve models without referring directly to a run ID or physical file path.

Later, the same model files were registered as version **2** so MLflow could serve them over HTTP to the Linux container. The `champion` alias was moved to version 2. This was not a newly trained model.

**In simple terms:**

**Logged model artifact -> output belonging to one training run**

**Registered model -> named and versioned model prepared for management and deployment**

---

## Question 2 - What replaced MLflow stages, and why use versions and aliases?

MLflow replaced the old built-in stages such as `Staging` and `Production` with flexible aliases.

Aliases are user-defined names such as:

* `champion` - the model currently selected for use.
* `challenger` - another model being tested against the champion.

A model is versioned separately from its training run because training and deployment are different activities. The run records how the model was created, while the registered version identifies the model that can be managed and deployed.

An alias is a movable pointer. For example, if a newer version performs better, the `champion` alias can be moved to it. The API can continue loading `models:/food11@champion` without changing its code.

**In simple terms:**

**Version -> permanent numbered model**

**Alias -> movable name pointing to the selected version**

---

## Question 3 - Why load the model using an MLflow model URI?

The API loads the model using:

`models:/food11@champion`

instead of using the direct path to a `.pth` file.

The MLflow URI hides the model's physical location and identifies it using a registered model name and alias. It also keeps the model connected to its MLflow metadata and version history.

To serve a newer model, we register it as another version and move the `champion` alias to that version. The API code does not need to change because it still loads `food11@champion`.

The running API must be restarted so that it loads the newly selected champion.

**In simple terms: the API follows the `champion` alias instead of depending on one fixed model file.**

---

## Question 4 - Why copy the dependency files before the source code?

The Dockerfile copies `pyproject.toml` and `uv.lock` and installs the dependencies before copying the `src/` folder.

Docker caches the result of each build instruction. The dependency files normally change less frequently than the application source code.

If only a line in `serve.py` changes, Docker can reuse the cached dependency layer. It only needs to copy the updated source code and rebuild the later layers.

If the entire project were copied before running `uv sync`, every source-code change would invalidate the dependency layer and force Docker to reinstall all packages.

**In simple terms: separating dependencies from source code makes repeated builds much faster.**

---

## Question 5 - What is the size difference between the single-stage and multi-stage images?

The naive single-stage image had a disk usage of **2.13 GB**, while the multi-stage image had a disk usage of **1.99 GB**.

Therefore, the multi-stage image was approximately **140 MB**, or **6.6%**, smaller.

The `docker history` results showed that the largest layer was the Python virtual environment:

* Multi-stage virtual environment: approximately **1.43 GB**
* Single-stage `uv sync` layer: approximately **1.48 GB**
* The single-stage image also contained the `uv` installation layer of approximately **60.1 MB**

The multi-stage build excludes `uv`, its cache, and other builder-only files from the final runtime image. However, it cannot remove PyTorch, MLflow, or other libraries required to run the API.

**In simple terms: the builder tools are removed, but the large runtime libraries must remain.**

---

## Question 6 - What happens if `.dockerignore` is missing?

Without `.dockerignore`, Docker sends unnecessary files and folders to the build process. This makes the build context larger and can make builds slower.

Folders such as `data/`, `mlruns/`, `.git/`, and `.venv/` can contain many large files. They would also increase the image size if the Dockerfile used a broad instruction such as `COPY . .`.

Sending a folder to the Docker daemon does not automatically add it to the image. It only becomes part of the image if a `COPY` or `ADD` instruction includes it.

The most dangerous excluded folder is `.venv/`. The local virtual environment was created for Windows, while the container runs Linux. Copying the Windows environment into the Linux image could overwrite the correct environment and break the application.

The dataset and MLflow folders would mainly make the build unnecessarily large. The model is fetched from MLflow at runtime, so these files do not belong inside the image.

**In simple terms: `.dockerignore` keeps the build small, fast, and free from incompatible local files.**

---

## Question 7 - Why cannot the container use `127.0.0.1:5000`?

A Docker container has its own isolated network.

Inside the container, `127.0.0.1` refers to the container itself, not the Windows host computer. Since MLflow is running on Windows, the container would not find it at its own localhost address.

On Docker Desktop, `host.docker.internal` resolves to an address that allows the container to contact the host computer.

Therefore, the container uses:

`http://host.docker.internal:5000`

to reach the MLflow server.

**In simple terms: `127.0.0.1` means the container, while `host.docker.internal` means the host computer.**

---

## Question 8 - Does the model load after starting a new container?

Yes.

The original container was stopped and removed. A completely new container was then created from the same `food11-api:latest` image without rebuilding it.

The new container started successfully, downloaded `food11@champion` from MLflow, and returned a prediction:

* Category: `Fried food`
* Confidence: approximately `0.6134`

The Docker image contains the Python environment, dependencies, source code, and API startup command. The trained model itself is fetched from MLflow when the container starts.

This means the model can be changed by moving the `champion` alias and restarting the container without rebuilding the Docker image.

It also means the MLflow server must be available when the container starts.

**In simple terms: the application is baked into the image, while the selected model is downloaded at runtime.**

---

## Question 9 - What is missing before another machine can run the exact image?

The Dockerfile is stored in Git, but the locally built Docker image is not automatically uploaded anywhere.

Before another machine, CI runner, or Kubernetes cluster can pull the image, it must be tagged and pushed to a container registry such as Docker Hub or GitHub Container Registry.

A fixed version tag or image digest should be used instead of depending only on the mutable `latest` tag. This ensures that another machine runs the exact image that was tested.

Git stores the instructions used to build the image, while a container registry stores the built image itself.

**In simple terms: the image still needs to be published to a container registry with an identifiable version or digest.**