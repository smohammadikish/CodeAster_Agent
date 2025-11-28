FROM simvia/code_aster:17.2.26

# Run as root to install packages
USER root

# Install runtime libraries required by gmsh (X11 + OpenGL deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    libglu1-mesa \
    libgl1-mesa-glx \
    libxrender1 \
    libxft2 \
    libfontconfig1 \
    libxext6 \
    libxcursor1 \
    libxinerama1 \
    libxi6 \
 && rm -rf /var/lib/apt/lists/*

# Install official gmsh binary (MED-enabled)
RUN wget https://gmsh.info/bin/Linux/gmsh-4.10.2-Linux64.tgz -O /tmp/gmsh.tgz \
 && cd /opt && tar xzf /tmp/gmsh.tgz \
 && ln -s /opt/gmsh-4.10.2-Linux64/bin/gmsh /usr/local/bin/gmsh \
 && rm /tmp/gmsh.tgz

WORKDIR /solver
COPY run_solver.py .

ENTRYPOINT ["python3", "run_solver.py"]
