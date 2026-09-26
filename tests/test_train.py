import numpy as np

from train import WindowSampler


def test_sampler_covers_each_window_once_per_epoch_and_is_resumable():
    s = WindowSampler(n_tokens=10 * 8 + 1, seq_len=8, batch_size=2, seed=0)
    assert s.n_windows == 10
    epoch0 = [x for step in range(5) for x in s.starts(step)]
    assert sorted(epoch0) == list(range(0, 80, 8))
    # A fresh sampler (as after --resume) reproduces any step exactly.
    assert WindowSampler(81, 8, 2, seed=0).starts(3) == s.starts(3)
    assert np.all(np.array(s.starts(7)) <= 80 - 8)
