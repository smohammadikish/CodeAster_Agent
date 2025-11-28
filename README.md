---

# 🚀 Docker Architecture for Code_Aster + Gmsh Agent

This document explains the new **two-container system** used to run **Gmsh meshing** and **Code_Aster simulation** from inside the chat-based application.

---

## 📦 Overview

The application now uses **two separate Docker images**:

### **1) `app-image` – Main Agent**

* Runs the LangChain-based agent.
* Generates:

  * `model.geo`
  * `model.comm`
  * `model.export`
* Stores files inside `workspace/`.
* **Does not run Gmsh or Code_Aster.**
* Instead, it calls the solver container through Docker.

### **2) `solver-image` – Meshing + Simulation**

* Contains:

  * Gmsh CLI with full MED support
  * Code_Aster 17.2 solver
* Takes `model.geo` & `model.export` as input.
* Produces:

  * `model.med`
  * `model.mess`
  * `model.resu`
* Results are saved back into `workspace/`.

---

## 📁 Shared Workspace Folder

Both containers share a common directory:

```
workspace/
```

Mapping:

| Host Path    | app-image        | solver-image |
| ------------ | ---------------- | ------------ |
| `workspace/` | `/app/workspace` | `/workspace` |

The app generates input files.
The solver container writes output files.

---

## 🔧 Updated Tools Inside the App

### 🟦 `mesh_with_gmsh_tool`

This tool **no longer executes Gmsh internally**.

It now:

* Validates that `model.geo` exists
* Returns confirmation that meshing can proceed
* Actual meshing is performed **inside solver-image**

---

### 🟩 `run_code_aster_tool`

This tool now:

1. Validates `model.export`
2. Prepares a Docker command:

   ```bash
   docker run --rm \
     -v HOST_WORKSPACE:/workspace \
     solver-image \
     model.geo model.export
   ```
3. Runs both:

   * Gmsh meshing
   * Code_Aster simulation
4. Returns results to the agent, including:

   * Paths to `.med`, `.mess`, `.resu`
   * Partial stdout logs

This design fully separates *agent logic* from *solver execution*.

---

## 🌍 Required Environment Variables in app-image

The app container needs:

### `SOLVER_IMAGE`

Name of the solver container:

```env
SOLVER_IMAGE=solver-image
```

### `HOST_WORKSPACE`

Absolute host path to the workspace folder:

```env
HOST_WORKSPACE=/Users/yourname/project/workspace
```

The app uses these values to construct the solver command.

---

## 🐳 Docker Requirements

Because app-image launches other containers, it must:

### 1. Include Docker CLI

Installed via apt in the Dockerfile.

### 2. Mount host Docker socket

When running app-image, you must mount:

```
-v /var/run/docker.sock:/var/run/docker.sock
```

This allows the container to run other containers — a method known as **Docker-outside-of-Docker (DooD)**.

---

## 🧠 Why Two Containers?

### ❌ Old Single-Container Approach

* Very large image
* Installation conflicts (gmsh, aster, python deps)
* No ARM/M1 compatibility
* Hard to maintain & debug

### ✔ New Dual-Container Architecture

* Clean separation of concerns
* app-image stays lightweight
* solver-image can be replaced/upgraded independently
* Fully compatible with **M1/M2 ARM systems**
* Ideal for production & CI/CD

---

## ▶️ Running Solver Manually (for debugging)

You can manually test the solver:

```bash
docker run --rm -it \
  -v /absolute/path/to/workspace:/workspace \
  solver-image \
  model.geo model.export
```

Generated output files will appear in `workspace/`.

---

## 📘 Summary Table

| Container        | Purpose              | Contains           | What it does                                 |
| ---------------- | -------------------- | ------------------ | -------------------------------------------- |
| **app-image**    | Chat-based AI agent  | Python + LangChain | Generates GEO, COMM, EXPORT, triggers solver |
| **solver-image** | Meshing + simulation | Gmsh + Code_Aster  | Produces MED, MESS, RESU                     |

Communication is handled via:

* Shared workspace volume
* Docker CLI (`docker run …`)

---

## 📝 Notes

* Comments in code are now standardized in **English**.
* Gmsh options include MED 4.0 compatibility for Code_Aster.
* Code is modular and easy to extend to cloud or cluster execution later.

---
