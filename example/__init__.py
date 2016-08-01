#-*- coding: utf-8 -*-

import numpy as np
import qutip as q
import os,sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, '..', 'lib'))


"""
set of functions to create
- hamiltonian
- collapse operators
- expectation operator
- density matrix

:author: Jesse Hinrichsen
"""

def get_V_expected(dim, omega, eta):
    """
    voltage expectation operator
    """
    a  = q.destroy(dim)
    ad = q.create(dim)
    
    phi = np.sqrt(eta/float(2.0*omega))*(ad+a)
    u   = 1.0j*np.sqrt(omega/float(2.0*eta))*(ad-a)

    return eta*u

def get_rho(dim, T, omega):
    """
    diagonal density matrix at temperature T for equidistant eigenstates
    """
    n = 1.0/(np.exp(omega/T)-1.0)
    return q.thermal_dm(dim, n)

def get_cops(dim, omega, T_bath, gamma):
    """
    collapse operators for equidistant eigenvalues
    """
    a  = q.destroy(dim)
    ad = q.create(dim)

    if T_bath == 0:
        return [0.5*gamma*a]

    n_bath = 1.0/(np.exp(omega/T_bath)-1.0)

    return [np.sqrt(0.5*gamma*(n_bath + 1))*a, np.sqrt(0.5*gamma*n_bath)*ad]

def get_hamiltonian(dim, eta, omega):
    """
    linear hamiltonian with range (for mesolve-cuda)
    """
    a  = q.destroy(dim)
    ad = q.create(dim)
    
    phi = np.sqrt(eta/float(2.0*omega))*(ad+a)
    u   = 1.0j*np.sqrt(omega/float(2.0*eta))*(ad-a)

    H0 = omega*(ad*a + 0.5)
    Hp = -phi/eta

    return [H0, [Hp, 'I0 * cos(wp[range_1]*t)']]


def get_hamiltonian_qutip(dim, eta, omega):
    """
    linear hamiltonian for qutip
    """
    a  = q.destroy(dim)
    ad = q.create(dim)
    
    phi = np.sqrt(eta/float(2.0*omega))*(ad+a)
    u   = 1.0j*np.sqrt(omega/float(2.0*eta))*(ad-a)

    H0 = omega*(ad*a + 0.5)
    Hp = -phi/eta

    return [H0, [Hp, 'I0 * cos(wp*t)']]