from collections import Counter

import numpy as np

from tgdata.sampling import MultiDatasetSampler


def test_sampler_temperature():
    sizes = [900, 100]
    def share(temperature):
        sampler = MultiDatasetSampler(sizes, 4, 2000, temperature, seed=0)
        return Counter(b[0] < 900 for b in sampler)[True] / 2000
    assert abs(share(1.0) - 0.9) < 0.03
    assert abs(share(1e6) - 0.5) < 0.04


def test_sampler_batches_stay_in_one_dataset():
    for batch in MultiDatasetSampler([10, 20, 30], 8, 50, 2.0):
        owners = {int(np.searchsorted([10, 30, 60], i, side="right")) for i in batch}
        assert len(owners) == 1
