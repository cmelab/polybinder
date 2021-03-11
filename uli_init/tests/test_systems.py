import random

from uli_init import simulate
from uli_init.tests.base_test import BaseTest


class TestSystems(BaseTest):
    def test_pdi(self):
        pass

    def test_build_peek(self):
        for i in range(5):
            compound = simulate.build_molecule("PEEK", i + 1, para_weight=0.50)

    def test_build_pekk(self):
        for i in range(5):
            compound = simulate.build_molecule("PEKK", i + 1, para_weight=0.50)

    def test_para_weight(self):
        all_para = simulate.random_sequence(para_weight=1, length=10)
        all_meta = simulate.random_sequence(para_weight=0, length=10)
        assert all_para.count("para") == 10
        assert all_meta.count("meta") == 10
        random.seed(24)
        mix_sequence = simulate.random_sequence(para_weight=0.50, length=10)
        assert mix_sequence.count("para") == 4
        random.seed()
        mix_sequence = simulate.random_sequence(para_weight=0.70, length=100)
        assert 100 - mix_sequence.count("para") == mix_sequence.count("meta")

    def test_gaff(self):
        pass

    def test_opls(self):
        pass
