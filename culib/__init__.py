#-*- coding: utf-8 -*-
"""
file helper

:author Jesse Hinrichsen
"""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
def load_lib_file(relative_path):
    with open(os.path.join(BASE, relative_path), 'r') as content_file:
        return content_file.read()