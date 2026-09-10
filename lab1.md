# Lab 1

## Question 1 — What do the files created by `uv init` contain?

When we run `uv init`, it creates the basic files needed for a Python project:

* **`pyproject.toml`** — contains information about the project, such as its name, Python version, and dependencies.
* **`.python-version`** — tells the project which Python version to use.
* **`README.md`** — contains information and documentation about the project.
* **`src/lab/__init__.py`** — contains Python code and makes `lab` a Python package.
* **`.venv/`** — the virtual environment containing Python and the packages installed for the project.

In simple terms, these files create an organized Python project with its own environment and dependencies.

---

## Question 2 — What files are created by `dvc init` and what should be pushed?

Running `dvc init` creates files used to configure DVC:

* **`.dvc/config`** — contains DVC settings, such as the remote storage configuration.
* **`.dvc/.gitignore`** — tells Git to ignore DVC's cache and temporary files.
* **`.dvcignore`** — tells DVC which files or folders to ignore.
* **`.dvc/cache/`** — stores copies of the actual data locally.
* **`.dvc/tmp/`** — contains temporary files used by DVC.

We should push `.dvc/config`, `.dvc/.gitignore`, and `.dvcignore` to Git.

We should **not** push `.dvc/cache/` or `.dvc/tmp/`.

**In simple terms: Git tracks the small configuration files, while DVC manages the large data files.**

---

## Question 3 — Where are the credentials stored?

Because we used `--global`, the DVC credentials are stored on the user's computer, outside the Git repository.

On Windows, they are usually stored in:

`C:\Users\User\AppData\Local\iterative\dvc\config`

Other options are:

* **`--project`** — stores settings in `.dvc/config`.
* **`--local`** — stores settings in `.dvc/config.local` and is useful for secrets specific to one project.
* **`--global`** — stores settings for the current user.
* **`--system`** — stores settings for all users on the computer.

### Should credentials be pushed to GitHub?

No. Passwords and tokens should **never** be pushed to GitHub.

The remote URL can be shared, but passwords and tokens should stay in local or global configuration.

---

## Question 4 — What happened to `.gitignore`?

When we ran:

`dvc add data`

DVC automatically added:

`/data`

to `.gitignore`.

This tells Git to ignore the `data/` folder because DVC is now responsible for tracking it.

Therefore:

* **DVC tracks the actual dataset.**
* **Git tracks the small `data.dvc` pointer file.**

This prevents thousands of large image files from being stored directly in Git.

---

## Question 5 — What is the `.dvc` file?

Running:

`dvc add data`

creates:

`data.dvc`

It is a small file containing information about the dataset, such as:

* **`md5`** — a hash identifying the version of the data.
* **`size`** — the size of the dataset.
* **`nfiles`** — the number of files.
* **`path`** — the folder being tracked.

The important idea is that **`data.dvc` acts like a pointer to a specific version of the dataset.**

Git tracks `data.dvc`, while DVC manages the actual large data files.

When someone needs the data, they can run:

`dvc pull`

DVC reads `data.dvc` and downloads the correct version of the dataset.

---

## Question 6 — GitHub Main Branch

### Is the code on GitHub?

Yes. The code and project files are stored on GitHub, such as `src/`, `pyproject.toml`, `README.md`, and the DVC configuration files.

### Is the data on GitHub?

No. The actual `data/` folder is not stored on GitHub because it is ignored by Git and managed by DVC.

### Is there a file that points to the data?

Yes:

`data.dvc`

It contains information that allows DVC to identify the correct version of the dataset.

### Is the data visible on DagsHub?

Yes. The actual dataset is uploaded to the DVC remote on DagsHub using:

`dvc push`

**In simple terms:**

**GitHub → code + `data.dvc` pointer**

**DagsHub → actual dataset**

---

## Question 7 — Fresh Clone

### Do we see the `data/` folder after cloning?

No.

When we run:

`git clone`

Git downloads the files stored in the Git repository, but the actual dataset is managed separately by DVC.

We get the `data.dvc` pointer, but not the actual `data/` folder.

### How do we get the data?

We run:

`dvc pull`

DVC reads `data.dvc`, downloads the correct dataset from the remote storage, and recreates the `data/` folder.

**In simple terms:**

**`git clone` → gets the code and DVC pointer**

**`dvc pull` → gets the actual data**

---

## Question 8 — Do you still see `food11_processed` and `food11_processed_mini`?

No.

After running:

`git checkout 761eea7`

and:

`dvc checkout`

the `data/` folder contains only:

`food11_raw`

The folders:

* `food11_processed`
* `food11_processed_mini`

are no longer there.

### Why?

The old Git commit was created **before the processed datasets existed**.

`git checkout 761eea7` restores the old version of the `data.dvc` pointer.

Then:

`dvc checkout`

makes the actual `data/` folder match that old pointer.

So the processed folders disappear because they were not part of that older version.

### What does this demonstrate?

Git and DVC work together to restore old versions:

**`git checkout` → restores the old code and `data.dvc` pointer**

**`dvc checkout` → restores the matching old data**

To return to the newest version:

`git checkout main`

`dvc checkout`

This brings the project back to the latest code and data.