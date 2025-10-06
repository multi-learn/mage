# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2025
# -----------------
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2020 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
#
# Description:
# -----------
#
#
#
# Version:
# -------
#
# * mage-multi-learn version = 1.0.0
#
# Licence:
# -------
#
# License: New BSD License
#
# ######### COPYRIGHT #########
import unittest
import os

from ..gaussian_classes import MultiViewGaussianSubProblemsGenerator

tmp_path = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    "tmp_tests", "")

def rm_tmp(path=tmp_path):
    try:
        for file_name in os.listdir(path):
            if os.path.isdir(os.path.join(path, file_name)):
                rm_tmp(os.path.join(path, file_name))
            else:
                os.remove(os.path.join(path, file_name))
        os.rmdir(path)
    except BaseException:
        pass

class Test_MultiViewGaussianSubProblemsGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_simple(self):
        gene = MultiViewGaussianSubProblemsGenerator(sub_problem_generators=["StumpsGenerator",
                                                                             "RingsGenerator",
                                                                             "StumpsGenerator",
                                                                             "RingsGenerator"])
        data, labels = gene.generate_multi_view_dataset()

    def test_report(self):
        gene = MultiViewGaussianSubProblemsGenerator(sub_problem_generators=["StumpsGenerator",
                                                                             "RingsGenerator",
                                                                             "StumpsGenerator",
                                                                             "RingsGenerator"])
        data, labels = gene.generate_multi_view_dataset()
        rep = gene.gen_report(save=False)

    def test_save(self):
        gene = MultiViewGaussianSubProblemsGenerator()
        data, labels = gene.generate_multi_view_dataset()
        rm_tmp()
        os.mkdir(tmp_path)
        gene.to_hdf5_mc(tmp_path)
        rep = gene.gen_report(output_path=tmp_path, save=True)
        rm_tmp()