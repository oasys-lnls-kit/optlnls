#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 08:37:49 2021

@author: lordano
"""

import numpy as np



def replace_char(filename, char, new_char, new_filename=''):
    
    with open(filename, 'r') as f:
        data = f.read()
        f.close()
        
    data = data.replace(char, new_char)
        
    if(new_filename == ''):
        new_filename = filename
    
    with open(new_filename, 'w') as f:
        f.write(data)
        f.close()