"""S4: recovery-caveat epsilon logic vs the H2 false-negative case."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.v2_art_loop import flag_recovery_caveats  # noqa: E402

# The exact colorable_light_raw_output_magnitude values from the committed H2
# artifact (out/h2-v2-artloop/report.json). Hand-checking that run showed 5/6
# genuine recoveries -- `practical` (max 1.59e-6, three orders below the eps-scale
# of the others) kept its neutral guess -- while the old fixed 1e-6 epsilon
# reported 6/6.
H2_MAGNITUDES = {
    "key": {"mean": 0.026946792379021645, "max": 1.4556833505630493},
    "fill": {"mean": 0.015473440289497375, "max": 1.2950308322906494},
    "rim": {"mean": 0.006399722769856453, "max": 1.1025245189666748},
    "window": {"mean": 0.00805407389998436, "max": 0.9233778119087219},
    "ceiling_panel": {"mean": 0.0044118482619524, "max": 0.3848559856414795},
    "practical": {"mean": 9.3143084356484e-09, "max": 1.5921024214549107e-06},
}


class RecoveryCaveatTests(unittest.TestCase):
    def test_h2_false_negative_now_flagged(self):
        caveats = flag_recovery_caveats(H2_MAGNITUDES)
        self.assertEqual([c["light"] for c in caveats], ["practical"])
        # 6 colorable lights, 1 caveat -> the automated count now matches the
        # hand-checked 5/6.
        self.assertEqual(len(H2_MAGNITUDES) - len(caveats), 5)

    def test_genuinely_contributing_lights_not_flagged(self):
        strong = {k: v for k, v in H2_MAGNITUDES.items() if k != "practical"}
        self.assertEqual(flag_recovery_caveats(strong), [])

    def test_absolute_floor_retained_for_all_zero_rigs(self):
        # every proxy dead: relative scaling must not mask the absolute criterion
        dead = {name: {"mean": 0.0, "max": 0.0} for name in ("a", "b")}
        self.assertEqual({c["light"] for c in flag_recovery_caveats(dead)}, {"a", "b"})

    def test_uniformly_weak_but_nonzero_rig_uses_absolute_floor(self):
        # all proxies equally weak-but-real (max well above 1e-6, no outlier):
        # nothing is flagged -- the relative test compares within the rig only.
        weak = {name: {"mean": 1e-4, "max": 5e-3} for name in ("a", "b", "c")}
        self.assertEqual(flag_recovery_caveats(weak), [])

    def test_empty_input(self):
        self.assertEqual(flag_recovery_caveats({}), [])


if __name__ == "__main__":
    unittest.main()
