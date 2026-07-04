import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.out_of_core import (  # noqa: E402
    cache_segment_bytes,
    stream_shard_targets,
    train_image_proxy_monolithic,
    train_image_proxy_streamed,
)
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
            mono_proxy, _ = train_image_proxy_monolithic(streamed, epochs=3, lr=0.5)
            streamed_proxy, opt_stats = train_image_proxy_streamed(
                shard_dir, lights, epochs=3, lr=0.5
            )
        mono = sum(gather_light(cache, light) for light in lights) / len(lights)
        np.testing.assert_allclose(streamed, mono, atol=1e-12)
        np.testing.assert_allclose(streamed_proxy, mono_proxy, atol=1e-12)
        self.assertLess(stats["stream_peak_segments_loaded"], cache.segment_count)
        self.assertLess(stats["stream_peak_segment_bytes_loaded"], cache_segment_bytes(cache))
        self.assertLess(opt_stats["streamed_optimizer_peak_segments_loaded"], cache.segment_count)
        self.assertLess(
            opt_stats["streamed_optimizer_peak_segment_bytes_loaded"], cache_segment_bytes(cache)
        )
        self.assertGreater(stats["stream_peak_shard_file_bytes"], 0)
        self.assertGreater(stats["stream_process_rss_after_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
