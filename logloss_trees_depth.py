# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:33:08 2016

@author: ryank
"""

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 


z = []

z = {'trees' : [11.76, 11.27, 11.24, 11.19,11.16, 11.14, 11.18],'depth' : [11.90, 11.14, 11.32, 11.31, 11.31,11.31, 11.31 ]}

y = pd.DataFrame(z)
x= [1,10, 20, 60, 80, 90, 100]

plt.plot(x,y['trees'], 'r', label = 'Number of Trees',  linewidth = 2.0)
plt.plot(x,y['depth'], label = 'Depth',  linewidth = 2.0)
plt.xlabel('Number of trees/Tree depth')
plt.legend('Upper left')
plt.ylabel('logloss')
plt.legend()

