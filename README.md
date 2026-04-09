# Distributed Contrast Stretching

A parallel image processing application that performs contrast stretching on bitmap images using MPI (Message Passing Interface). The algorithm makes lighter pixels lighter and darker pixels darker by analyzing each pixel's nearest neighbors — distributed across multiple processes for high performance.

Built for and deployed on Northwestern University's **Quest HPC Cluster** (1,000+ node supercomputer), achieving near-linear speedup when scaling across multiple nodes.

---

## How It Works

Contrast stretching is applied iteratively. For each pixel, the algorithm:

1. Examines the pixel's 8 surrounding neighbors
2. Computes the median, min, and max of those neighbors
3. If the pixel is relatively darker than its neighbors → decrease its value
4. If the pixel is relatively lighter than its neighbors → increase its value
5. Repeats for a given number of steps, or until the image converges (no more changes)

### MPI Parallelism

The image is split into horizontal chunks and distributed across processes using `MPI_Scatter`. Each process independently stretches its chunk, exchanging **ghost rows** with neighboring processes at each step to maintain correctness at chunk boundaries. Results are collected back to the master process via `MPI_Gather`.

Convergence is detected globally using `MPI_Reduce` to sum pixel differences across all processes, followed by `MPI_Bcast` to notify everyone.

```
Master Process                Worker Processes
──────────────                ────────────────
Read .bmp file
Distribute chunks  ────────>  Receive chunk
                              Exchange ghost rows (each step)
                              Stretch pixels
Collect chunks     <────────  Send chunk back
Write output .bmp
```

---

## Project Structure

```
distributed-contrast-stretching/
├── mpi/               # Parallel MPI implementation
│   ├── main.cpp       # Entry point: distribute, stretch, collect
│   ├── cs.cpp         # Contrast stretch logic (ContrastStretch, NewPixelValue)
│   ├── bitmap.cpp     # BMP file read/write
│   ├── app.h          # Shared types and function declarations
│   ├── matrix.h       # 2D matrix allocation helpers
│   └── makefile
├── seq/               # Sequential (single-process) baseline implementation
│   ├── main.cpp
│   ├── cs.cpp
│   └── makefile
├── docker/            # Docker environment for local development
│   ├── Dockerfile
│   ├── build.bash / build.bat / build.ps1
│   └── run.bash / run.bat / run.ps1
└── ReferenceFiles/    # Original starter code for reference
```

---

## Usage

### Command Line

```bash
mpiexec -n <num_processes> ./cs infile.bmp outfile.bmp steps
```

| Argument | Description |
|---|---|
| `infile.bmp` | Input Windows bitmap image |
| `outfile.bmp` | Output file path for the processed image |
| `steps` | Max number of stretch iterations (stops early if converged) |

**Example:**
```bash
mpiexec -n 8 ./cs sunset.bmp stretched-output.bmp 75
```

### Building

```bash
cd mpi
make
```

For the sequential version:
```bash
cd seq
make
```

---

## Running with Docker (Local Development)

Docker is provided for running MPI locally without installing it manually.

```bash
cd docker

# Build the image:
./build.bash       # Mac/Linux
./build.ps1        # Windows PowerShell

# Run the container:
./run.bash         # Mac/Linux
./run.ps1          # Windows PowerShell
```

---

## Running on Quest HPC (Northwestern)

This project was developed and benchmarked on Northwestern's Quest supercomputer.

Submit a job with your desired node/process count:

```bash
mpiexec -n <num_processes> ./cs infile.bmp outfile.bmp steps
```

Scaling was tested from 1 to 16 nodes, achieving near-linear speedup.

---

## Requirements

- C++ compiler with MPI support (`mpic++`)
- MPI library (e.g., OpenMPI or MPICH)
- Input must be a 24-bit Windows Bitmap (`.bmp`) file

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | C++ |
| Parallelism | MPI (OpenMPI / MPICH) |
| Image Format | Windows Bitmap (.bmp) |
| HPC Platform | Northwestern Quest Cluster |
| Containerization | Docker |

---

## Authors

- **Anthony Milas** — MPI parallelization, convergence detection, ghost row communication
- **Prof. Joe Hummel** — Original sequential algorithm (Northwestern University)
