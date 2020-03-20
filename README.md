# WIP Multiview Generator

This package aims at generating mutliview datasets according to multipe paramterers : 
an error matrix, that tracks the error of each view on each class, a redundancy float that controls the ration of examples that are well described by all the viewsw, 
similarly the lutual error float controls the one that are misdescribed in eavery views.

## Structure    

The class of intereset is located in ``generator/multiple_sub_problems.py`` and called ``MultiViewSubProblemsGenerator``. 

A demo is available in ``demo/demo.py`` and generates a 3D dataset, along with a figure that analyzes it.   