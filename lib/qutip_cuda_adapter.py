#-*- coding: utf-8 -*-

"""
adapter to work qutip-like

:author Jesse Hinrichsen
"""

import qutip as q
import numpy as np

from lib.cuda_solver import MesolveResult

class QtipMesolveResult(MesolveResult):
    def __init__(self, hamiltonian, rho, tlist, c_ops, e_ops, params):
        MesolveResult.__init__(self,
            self._extract_full(hamiltonian),
            self._extract_full(rho),
            tlist,
            self._extract_full(c_ops),
            self._extract_full(e_ops),
            params
        )


    def _extract_full(self, obj):
        if isinstance(obj, list):
            _obj = []
            for o in obj:
                _obj.append(self._extract_full(o))
            return _obj
        elif isinstance(obj, q.Qobj):
            return obj.full() 
        elif isinstance(obj, str):
            return obj
        else:
            raise RuntimeError("Unexpected type {}".format(type(obj)))
        

def mesolve(hamiltonian, rho, tlist, c_ops, e_ops, params):
    def _verify_qobj(obj, allow_string=False):
        if isinstance(obj, list):
            for o in obj:
                _verify_qobj(o, allow_string)
        elif not (isinstance(obj, q.Qobj) or (allow_string and isinstance(obj, str))):
            raise TypeError("Expected {} or {} but got {}".format(q.Qobj, str, type(obj)))

    parallel_params = []
    for k, v in params:
        if isinstance(v, list):
            parallel_params.append(k)
    if len(parallel_params) > 2:
        raise ValueError("Got more than 2 parameter ranges: {}".format(parallel_params))

    _verify_qobj(hamiltonian, allow_string=True)
    _verify_qobj(rho)
    _verify_qobj(c_ops)
    _verify_qobj(e_ops)


    result = QtipMesolveResult(hamiltonian, rho, tlist, c_ops, e_ops, params)
    result.solve()
    return result


#def mesolve_advanced(hamiltonian, rho, tlist, c_ops, e_ops, params):
#    def _verify_qobj(obj, allow_string=False):
#        if isinstance(obj, list):
#            for o in obj:
#                _verify_qobj(o, allow_string)
#        elif not (isinstance(obj, q.Qobj) or (allow_string and isinstance(obj, str))):
#            raise TypeError("Expected {} or {} but got {}".format(q.Qobj, str, type(obj)))
#
#    parallel_params = []
#    for k, v in params:
#        if isinstance(v, list):
#            parallel_params.append(k)
#    if len(parallel_params) > 2:
#        raise ValueError("Got more than 2 parameter ranges: {}".format(parallel_params))
#
#    _verify_qobj(hamiltonian, allow_string=True)
#    _verify_qobj(rho)
#    _verify_qobj(c_ops)
#    _verify_qobj(e_ops)
#
#    result = QtipMesolveResult(hamiltonian, rho, tlist, c_ops, e_ops, params)
#    return result