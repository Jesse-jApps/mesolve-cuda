# mesolve-cuda
Based on qutip mesolve, a simple implementation using pycuda

## Usage
The same api as qutip
V_cuda = mesolve(hamiltonian, rho0, tlists, c_ops, e_ops, params)

- tlists is expected to be a list of tlist (a tlist for every possible parameter)
- params can hold parameter ranges:
params = [
    ('I0', I),
    ('wp', np.arange(0.1, 2.0, 0.1))
]

## Example
- volatage response for a single jospehson junction
- performance test

###Calculation of voltage with qutip and cuda
![Voltage](https://github.com/Jesse-jApps/mesolve-cuda/blob/master/voltage.png "Voltage")

###Performance
![Performance](https://github.com/Jesse-jApps/mesolve-cuda/blob/master/performance.png "Performance")


## Todos
- Better Runge-Kutta implementation - 5(4)
- Adaptive stepwidth
- Precompile matrix multiplication for sparse operations

## Notes
To enable use cuda devices, the correct environment has to be set

export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-7.5/lib64

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64

export C_INCLUDE_PATH=$C_INCLUDE_PATH:/usr/local/cuda-7.5/include

export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/local/cuda-7.5/include

export PATH=$PATH:/usr/local/cuda-7.5/bin