#-*- coding: utf-8 -*-

"""
mesolve implementation using pycuda

:author Jesse Hinrichsen
"""

import numpy as np
from jinja2 import Template
from . import load_lib_file

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

class MesolveResult():
    """docstring for MesolveResult"""
    def __init__(self, hamiltonian, rho, tlist, c_ops, e_ops, params):
        self.rho         = rho
        self.tlist       = tlist
        self.expectation_length = len(tlist[0])
        self.c_ops       = c_ops
        self.c_ops_dagger = [c.T for c in self.c_ops]
        self.e_ops       = e_ops
        self.expect      = []

        free, total = drv.mem_get_info()
        print("0", free/float(total))


        self.params       = []
        self.range_params = []
        for k,v in params:
            if isinstance(v, np.ndarray):
                self.range_params.append((k, v))
            else:
                self.params.append((k,v))
        if len(self.range_params) > 2:
            raise ValueError("More than 2 range parameters")

        self.block = [1,1,1]
        self.grid  = (1,1)
        if len(self.range_params) == 2:
            range_1 = len(self.range_params[0][1])
            range_2 = len(self.range_params[1][1])
            if range_1*range_2 > self._max_number_of_threads():
                raise ValueError("Too many parameters in ranges")
            self.block[0] = range_1
            self.block[1] = range_2
        elif len(self.range_params) == 1:
            range_1 = len(self.range_params[0][1])
            if range_1 > self._max_number_of_threads():
                raise ValueError("Too many parameters in range")
            self.block[0] = range_1
        self.block = tuple(self.block)


        self.independant_hamiltonian = []
        self.dependant_hamiltonian = []
        if isinstance(hamiltonian, list):
            if isinstance(hamiltonian, np.ndarray) and isinstance(hamiltonian[1], str):
                self.independant_hamiltonian.append(hamiltonian)
            else:
                for h in hamiltonian:
                    if isinstance(h, list):
                        if not (isinstance(h[0], np.ndarray) and isinstance(h[1], str)):
                            raise ValueError("Unknown time-dependant hamiltonian type")
                        self.dependant_hamiltonian.append(h)
                    elif isinstance(h, np.ndarray):
                        self.independant_hamiltonian.append(h)
                    else:
                        raise ValueError("Unknown time-independant hamiltonian type")
        else:
            self.independant_hamiltonian.append(hamiltonian)
        
        self.dim = rho.shape[0]
        

        KERNEL = load_lib_file("mesolve_kernel.c")

        precompiled_kernel = Template(KERNEL).render(**
        {
            'dim': self.dim,
            'mat_size': self.dim**2,
            'range_1_dim': self.block[0],
            'total_range_size': self.block[0]*self.block[1],
            #'step_size': np.float32(self.tlist[1] - self.tlist[0]),
            'steps': self.expectation_length,
            'num_c_ops': len(self.c_ops),
            'h0_args': self._construct_arguments(self.independant_hamiltonian,'const dcmplx *', 'h_ind'),
            'h0_commutators_k1': self._construct_k1_commutator(self.independant_hamiltonian, 'h_ind', diag=True),
            'h0_commutators_k2': self._construct_k2_commutator(self.independant_hamiltonian, 'h_ind', diag=True),
            'h0_commutators_k3': self._construct_k3_commutator(self.independant_hamiltonian, 'h_ind', diag=True),

            'ht_args': self._construct_arguments(self.dependant_hamiltonian, 'const dcmplx *', 'h_dep'),
            'ht_commutators_k1': self._construct_k1_commutator(self.dependant_hamiltonian, 'h_dep'),
            'ht_commutators_k2': self._construct_k2_commutator(self.dependant_hamiltonian, 'h_dep'),
            'ht_commutators_k3': self._construct_k3_commutator(self.dependant_hamiltonian, 'h_dep'),

            'params': self._construct_arguments(self.params, 'double '), 
            'range_params': self._construct_arguments(self.range_params, 'const dcmplx *'), 

            'e_ops': self._construct_arguments(self.e_ops, 'const dcmplx *', 'e'),
            'expects': self._construct_arguments(self.e_ops, 'dcmplx *', 'expect'),
            'expectations': self._construct_expectation(self.e_ops, self.expectation_length, 'e')
        })

        #print(precompiled_kernel)
        #print(self.block)
        #sdf()

        program = SourceModule(precompiled_kernel)
        self.mesolve = program.get_function("mesolve")

    def _construct_expectation(self, e_ops, length, base_name):
        code = []
        for i, e in enumerate(e_ops):
            code.append('expect_{}[range_offset*{}+i] = matrix_trace(rho, {}_{}, range_matrix_offset, 0);'.format(i, length, base_name, i))
        return code

    def _construct_arguments(self, buffers, data_type, base_name=""):
        args = []
        for i, b in enumerate(buffers):
            if isinstance(b[0], str):
                args.append('{}{}'.format(data_type, b[0]))
            else:
                args.append('{}{}_{}'.format(data_type, base_name, i))

        return ', '.join(args)

    def _construct_k1_commutator(self, buffers, base_name, diag=False):
        code = []
        for i, b in enumerate(buffers):
            factor = "1.0"
            if isinstance(b[1], str):
                factor = b[1]
            if diag:
                code.append('matrix_dot_add_existing_diag_left({}_{}, rho, k1, 0, range_matrix_offset, 0, -1.0*imag*{});'.format(base_name, i, factor))
                code.append('matrix_dot_add_existing_diag_right(rho, {}_{}, k1, range_matrix_offset, 0, 0, 1.0*imag*{});'.format(base_name, i, factor))
            else:
                code.append('matrix_dot_add_existing({}_{}, rho, k1, 0, range_matrix_offset, 0, -1.0*imag*{});'.format(base_name, i, factor))
                code.append('matrix_dot_add_existing(rho, {}_{}, k1, range_matrix_offset, 0, 0, 1.0*imag*{});'.format(base_name, i, factor))

        return code


    def _construct_k2_commutator(self, buffers, base_name, diag=False):
        code = []
        for i, b in enumerate(buffers):
            factor = "1.0"
            if isinstance(b[1], str):
                factor = b[1]
            if diag:
                code.append('matrix_dot_add_existing_diag_left({}_{}, tmp, k2, 0, 0, 0, -1.0*imag*{});'.format(base_name, i, factor))
                code.append('matrix_dot_add_existing_diag_right(tmp, {}_{}, k2, 0, 0, 0, 1.0*imag*{});'.format(base_name, i, factor))
            else:
                code.append('matrix_dot_add_existing({}_{}, tmp, k2, 0, 0, 0, -1.0*imag*{});'.format(base_name, i, factor))
                code.append('matrix_dot_add_existing(tmp, {}_{}, k2, 0, 0, 0, 1.0*imag*{});'.format(base_name, i, factor))

        return code

    def _construct_k3_commutator(self, buffers, base_name, diag=False):
        code = []
        for i, b in enumerate(buffers):
            factor = "1.0"
            if isinstance(b[1], str):
                factor = b[1]
            if diag:
                code.append('matrix_dot_add_existing_diag_left({}_{}, tmp, k3, 0, 0, 0, -1.0*imag*{});'.format(base_name, i, factor))
                code.append('matrix_dot_add_existing_diag_right(tmp, {}_{}, k3, 0, 0, 0, 1.0*imag*{});'.format(base_name, i, factor))
            else:
                code.append('matrix_dot_add_existing({}_{}, tmp, k3, 0, 0, 0, -1.0*imag*{});'.format(base_name, i, factor))
                code.append('matrix_dot_add_existing(tmp, {}_{}, k3, 0, 0, 0, 1.0*imag*{});'.format(base_name, i, factor))

        return code


    def _max_number_of_threads(self):
        return pycuda.tools.DeviceData().max_threads


    def solve(self):
        free, total = drv.mem_get_info()
        print("1", free/float(total))

        h0 = []
        for h in self.independant_hamiltonian:
            h0.append(drv.In(h.astype(np.complex128)))

        ht = []
        for h in self.dependant_hamiltonian:
            ht.append(drv.In(h[0].astype(np.complex128)))


        params = []
        for k, v in self.params:
            params.append(np.float64(v))


        range_params = []
        for k, v in self.range_params:
            range_params.append(drv.In(v.astype(np.complex128)))

        step_sizes = []
        for t in self.tlist:
            step_sizes.append(t[2] - t[1])
        #step_sizes = gpuarray.to_gpu(np.array(step_sizes, dtype=np.float64))
        step_sizes = drv.In(np.array(step_sizes, dtype=np.float64))


        #print(step_sizes)
        #sdf()

        #c_ops        = drv.In(np.array(self.c_ops, dtype=np.complex128))
        c_ops        = gpuarray.to_gpu(np.array(self.c_ops, dtype=np.complex128))
        c_ops_dagger = gpuarray.to_gpu(np.array(self.c_ops_dagger, dtype=np.complex128))

        rho = np.array([self.rho for i in range(self.block[0]*self.block[1])], dtype=np.complex128)
        rho = gpuarray.to_gpu(rho)

        e_ops = []
        for e in self.e_ops:
            e_ops.append(drv.In(e.astype(np.complex128)))

        expects = []
        for e in self.e_ops:
            expects.append(gpuarray.to_gpu(np.zeros((self.block[0]*self.block[1]*self.expectation_length)).astype(np.complex128)))

        arguments = h0 + ht + [c_ops, c_ops_dagger, step_sizes, rho] + params + range_params + e_ops + expects

        #print(arguments)
        #sdf()

        #print(rho.get())

        self.mesolve(*arguments,
            block=self.block,
            grid=self.grid
        )

        free, total = drv.mem_get_info()
        print("2", free/float(total))

        del rho
        del c_ops
        del c_ops_dagger

        print(self.block)
        for e in expects:
            self.expect.append(np.array(e.get()).reshape(self.block[0], self.block[1], self.expectation_length))
            del e

        

        free, total = drv.mem_get_info()
        print("3", free/float(total))