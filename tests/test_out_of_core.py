import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.out_of_core import stream_shard_targets  # noqa: E402
from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


class OutOfCoreTests(unittest.TestCase):
    def test_streamed_targets_match_monolithic_gather(self):
        cache = trace_path_cache(12, 12, 4, 2, seed=17)
        lights = [
            SphereLight(center=[0.1, 0.6, 0.0], radius=0.2, rgb=[1.5, 1.0, 0.75]),
            SphereLight(center=[0.75, 0.75, 0.35], radius=0.12, rgb=[0.8, 1.2, 1.0]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = Path(tmp) / "shards"
            cache.save_sharded(str(shard_dir), tile_size=4)
            streamed, stats = stream_shard_targets(shard_dir, lights)
        mono = sum(gather_light(cache, light) for light in lights) / len(lights)
        np.testing.assert_allclose(streamed, mono, atol=1e-12)
        self.assertLess(stats["stream_peak_segments_loaded"], cache.segment_count)


if __name__ == "__main__":
    unittest.main()
