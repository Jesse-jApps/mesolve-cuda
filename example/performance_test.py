#-*- coding: utf-8 -*-
"""
performance test

:author: Jesse Hinrichsen
"""

import qutip as q
import numpy as np

from example import get_hamiltonian, get_hamiltonian_qutip, get_cops, get_rho, get_V_expected

import time

from lib.qutip_cuda_adapter import mesolve
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dim = 5
    eta = 1
    omega = 1
    gamma = 0.3
    T = 0.1
    I = 0.01

    driving_periods  = 40
    steps_per_period = 150

    steps = steps_per_period * driving_periods

    step_sizes = np.arange(0.02, 0.2, 0.04)[::-1]
    performance_result = np.zeros((2, len(step_sizes)))
    number_of_steps = np.zeros(len(step_sizes))

    for si in range(len(step_sizes)):
        step_size = step_sizes[si]
        print("stepsize: {}".format(step_size))

        wp_range = np.arange(0.1, 2.0, step_size)
        number_of_steps[si] = len(wp_range)

        print("number of parameters: {}".format(len(wp_range)))

        h_t       = get_hamiltonian(dim, eta, omega)
        h_t_qutip = get_hamiltonian_qutip(dim, eta, omega)

        rho0  = get_rho(dim, T, omega)
        c_ops = get_cops(dim, omega, T, gamma)

        v_expectation = get_V_expected(dim, omega, eta)

        tlists = []
        for wpi, wp in enumerate(wp_range):
            h = (2*np.pi / wp) / steps_per_period
            tlists.append(np.array([x*h for x in range(steps)]))
        #cuda implementation
        start_time = time.time()
        V_cuda = mesolve(h_t, rho0, tlists, c_ops, [v_expectation], [
            ('I0', I),
            ('wp', wp_range)
        ]).expect[0]


        V_result_cuda = np.zeros(len(wp_range), dtype=np.complex128)
        for wpi, wp in enumerate(wp_range):
            h = (2*np.pi / wp) / steps_per_period

            V = V_cuda[wpi][0]

            relevant_data = V
            ft = [x*np.exp(1.0j*wp*h*z) for z, x in enumerate(relevant_data)]
            summe = np.sum(ft)
            
            V_result_cuda[wpi] = summe / len(relevant_data)
            #print(V_result_cuda[wpi])

        exec_time = time.time() - start_time
        print("cuda: {}".format(exec_time))
        performance_result[0][si] = exec_time

        #qutip implementation
        start_time = time.time()
        V_result_qutip = np.zeros(len(wp_range), dtype=np.complex128)
        for wpi, wp in enumerate(wp_range):
            h     = (2*np.pi / wp) / steps_per_period
            tlist = np.array([x*h for x in range(steps)])
            print(wp, h)

            V_qutip = q.mesolve(h_t_qutip, rho0, tlist, c_ops, [v_expectation], {'I0': I, 'wp': wp}).expect[0]

            relevant_data = V_qutip
            ft = [x*np.exp(1.0j*wp*h*z) for z, x in enumerate(relevant_data)]
            summe = np.sum(ft)

            V_result_qutip[wpi] = summe / len(relevant_data)
            #print(V_result_qutip[wpi])

        exec_time = time.time() - start_time
        print("qutip: {}".format(exec_time))
        performance_result[1][si] = exec_time


    fig, axes = plt.subplots(1,1)
    axes.plot(number_of_steps, performance_result[0])
    axes.plot(number_of_steps, performance_result[1])

    axes.set_xlabel(r'number of params', fontsize=20)
    axes.set_ylabel(r"time", fontsize=16);
    plt.savefig('performance.pdf')
