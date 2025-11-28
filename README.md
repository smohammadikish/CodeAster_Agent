---

# 🚀 Docker Architecture for Code_Aster + Gmsh Agent

This document describes the **two-container architecture** used for running **Gmsh meshing** and **Code_Aster simulation** from inside the LangChain-based agent.

---

# 🏗 Architecture Overview

The system consists of **two Docker images** working together:

---

## **1) `app-image` – Main Application (LangChain Agent)**

Responsible for:

* Running the Chat/LLM agent
* Generating input files:

  ```
  model.geo
  model.comm
  model.export
  ```
* Managing the workspace
* Calling the solver container via Docker

Does **not** run Gmsh or Code_Aster internally.

---

## **2) `solver-image` – Mesher + FEA Solver**

Contains:

* **Gmsh CLI** (compiled with MED support)
* **Code_Aster 17.2 solver**

Responsible for:

* Reading `model.geo`
* Generating `model.med`
* Running Code_Aster using `model.export`
* Producing:

  ```
  model.med
  model.mess
  model.resu
  model.rmed (if applicable)
  ```

All results are saved into the shared `workspace/` folder.

---

# 📁 Shared Workspace Design

Both containers use **the same folder**:

```
workspace/
```

Mapping:

| Host         | app-image        | solver-image |
| ------------ | ---------------- | ------------ |
| `workspace/` | `/app/workspace` | `/workspace` |

### Workflow:

1. app generates: `.geo`, `.comm`, `.export`
2. solver reads them, runs Gmsh + Code_Aster
3. solver writes back `.med`, `.mess`, `.resu`
4. app parses the results

---

# 🔧 Updated Tools in the App

The agent tools no longer execute binaries locally.
They **prepare files and trigger the solver**.

---

## 🟦 `mesh_with_gmsh_tool`

✔ Validates that `model.geo` exists
✔ No longer runs Gmsh
✔ Allows agent to proceed to solver call

Meshing now always happens inside `solver-image`.

---

## 🟩 `run_code_aster_tool`

Runs everything through Docker:

1. Ensures `model.export` exists
2. Builds a Docker command:

```bash
docker run --rm \
  -v HOST_WORKSPACE:/workspace \
  ${SOLVER_IMAGE} \
  model.geo model.export
```

3. Executes:

* Gmsh (meshing)
* Code_Aster (simulation)

4. Returns:

* paths of `.med`, `.mess`, `.resu`
* partial stdout logs
* any error messages

This isolates heavy solvers from the lightweight application image.

---

# 🌍 Required Environment Variables

A **single `.env` file** lives in the project root (next to docker-compose.yml).

### Example `.env`:

```env
# Solver image name
SOLVER_IMAGE=solver-image

# Absolute path to workspace on host
HOST_WORKSPACE=/Users/yourname/training/workspace

# Agent keys
OPENAI_API_KEY=xxxx
GROQ_API_KEY=xxxx
ANTHROPIC_API_KEY=xxxx

# Fallback paths (not used inside app now)
GMSH_BIN=gmsh
AS_RUN_BIN=/opt/spack/var/spack/environments/simvia_env/.spack-env/view/bin/run_aster
```

---

# 🐳 Docker Requirements for app-image

Because app-image needs to launch another container (solver-image), it must support **Docker-outside-of-Docker (DooD)**:

### 1) Docker CLI must be installed inside the image

Added in the app-image Dockerfile.

### 2) App container must mount the host’s Docker socket:

```
-v /var/run/docker.sock:/var/run/docker.sock
```

This allows:

* `docker run`
* `docker ps`
* `docker logs`
* etc.

from inside the app.

---

# 🐙 docker-compose.yml

The `docker-compose.yml`:

* Builds both images
* Mounts workspace to both containers
* Passes env vars
* Provides docker.sock to app-image

(نسخه کامل به زودی داخل همین پروژه قرار می‌گیرد.)

---

# ▶️ Manual Test of Solver Image

Debugging the solver without the app:

```bash
docker run --rm -it \
  -v /absolute/path/to/workspace:/workspace \
  solver-image \
  model.geo model.export
```

You should see:

* Gmsh meshing logs
* Code_Aster output log
* Files written in `workspace/`

---

# 📘 Summary Table

| Image          | Purpose                 | Includes           | Produces                                    |
| -------------- | ----------------------- | ------------------ | ------------------------------------------- |
| `app-image`    | AI agent, file creation | Python + LangChain | `.geo`, `.comm`, `.export`; triggers solver |
| `solver-image` | Meshing + FEA solver    | Gmsh + Code_Aster  | `.med`, `.mess`, `.resu`, maybe `.rmed`     |

Communication happens through:

* Shared workspace
* Docker CLI (`docker run`)

---

# 📝 Notes

* All code comments are standardized in **English**
* Gmsh MED version is forced to 4.0 for Code_Aster compatibility
* Infrastructure is cloud-ready (solver can run remotely)
* The solver-image (Code_Aster) is x86_64-only and runs on Apple Silicon via QEMU emulation, which may cause reduced performance or CPU instruction errors.

---
