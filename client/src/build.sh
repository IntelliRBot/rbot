#!/bin/bash

swig -c++ -python DcmAlgorithm.i 
python3.7 setup.py build_ext --inplace