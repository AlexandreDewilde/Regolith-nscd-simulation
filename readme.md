# NSCD simulation in 2D (and 3D) 🌟

## About
This tool provides a way to run NSCD (Non-Specific Cell Death) simulations. It's built in Python, utilizing Numba for code optimization. The GUI is developed using PyFX, a high-level computer graphics library based on wgpu-py with WebGPU support.

## Installation
To get started, follow these steps:

1. Clone the repository:
   ```shell
   git clone git@github.com:AlexandreDewilde/project-advanced.git
   ```

2. Navigate to the project directory:
   ```shell
   cd project-advanced
   ```

3. Install the required dependencies:
   ```shell
   pip install -r requirements.txt
   ```

## Example Run
To run an example simulation, execute the following command:
```shell
python main.py
```

Feel free to explore the fascinating world of NSCD simulations! 🧪🔬🌐

## API usage

### Simulation
1. Generate particle positions:
   ```python
   positions = insert_particles(n, 1/4, xmin, xmax, ymin, ymax)
   n = len(positions)
   ```

2. Initialize other properties:
   ```python
   velocities = np.zeros((n, 2))
   omega = np.zeros(len(positions))
   radius = np.ones(len(positions)) / 4
   dt = 1e-6
   g = 9.81
   ```

3. Create the `Simulation` instance:
   ```python
   sim = Simulation(
       init_positions=positions,
       init_velocities=velocities,
       init_omega=omega,
       radius=radius,
       rho=rho,
       g=g,
       d3=False,
       tree=True,
       precomputation_file="out.txt"
   )
   ```

### GUI

```python
sim = Simulation(.....)
gui = GUI(sim, bound=(0, 20, 0, 20))
```

Feel free to adapt this example to your specific use case! 🚀🔬🤖