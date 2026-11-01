"""Targeted negative controls for independent audits."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import unittest

from benchmarks import check_fcidump_h4, check_noise_aware_selection

ROOT = Path(__file__).resolve().parents[1]


class IndependentCheckTests(unittest.TestCase):
    def test_h4_independent_prefix_and_word_count(self):
        record = json.loads(check_fcidump_h4.RECORD.read_text())
        result = check_fcidump_h4.audit(record)
        self.assertEqual(result["selected_words"], 7371)
        broken = deepcopy(record)
        broken["adaptive_acase"]["energy_history"][3] += .01
        with self.assertRaisesRegex(AssertionError, "prefix energies"):
            check_fcidump_h4.audit(broken)
        broken = deepcopy(record)
        broken["adaptive_acase"]["word_universe"] -= 1
        with self.assertRaisesRegex(AssertionError, "word universe"):
            check_fcidump_h4.audit(broken)

    def test_noise_outcomes_and_source_list_cannot_be_silently_changed(self):
        record = json.loads(gzip.decompress(check_noise_aware_selection.RESULT.read_bytes()))
        self.assertEqual(check_noise_aware_selection.check(record), 4)
        check_noise_aware_selection.check_report(record)
        broken = deepcopy(record)
        broken["summaries"][0]["median_ratio"] = 0.9
        with self.assertRaisesRegex(AssertionError, "report table"):
            check_noise_aware_selection.check_report(broken)
        broken = deepcopy(record)
        del broken["source_sha256"]["clifford_qc/fermion.py"]
        with self.assertRaisesRegex(AssertionError, "source list"):
            check_noise_aware_selection.check(broken)
        broken = deepcopy(record)
        broken["outcomes"][0]["rows"][0]["training_counts"][0] ^= 1
        with self.assertRaisesRegex(AssertionError, "sampled counts"):
            check_noise_aware_selection.check(broken)


if __name__ == "__main__":
    unittest.main()
