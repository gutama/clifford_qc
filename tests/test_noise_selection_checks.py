"""Targeted negative controls for independent audits."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

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
        # Core CI and recorded-replay CI use different NumPy versions. This
        # mode checks the numerical results from the recorded draws; the pinned
        # replay gate also independently regenerates the seeded random stream.
        self.assertEqual(check_noise_aware_selection.check(record, replay_rng=False), 4)
        if np.__version__ == record["environment"]["numpy"]:
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
        with self.assertRaisesRegex(AssertionError, "training energy"):
            check_noise_aware_selection.check(broken, replay_rng=False)

    def test_a_bound_source_may_evolve_only_beside_its_packaged_execution_bytes(self):
        """Drift fails; a deliberate, packaged change is reported, not hidden."""
        checker = check_noise_aware_selection
        record = json.loads(gzip.decompress(checker.RESULT.read_bytes()))
        evolved = checker.evolved_sources(record)
        self.assertEqual(evolved, sorted(evolved))
        self.assertLessEqual(set(evolved), set(record["source_sha256"]))
        # Every packaged snapshot is the recorded bytes, whether or not the live
        # file has moved yet.
        for snapshot in sorted(p for p in checker.EXECUTION_SOURCES.rglob("*") if p.is_file()):
            relative = snapshot.relative_to(checker.EXECUTION_SOURCES).as_posix()
            self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                             record["source_sha256"][relative], relative)
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory)
            if evolved:
                # Without the snapshot, the same live tree is plain drift.
                with patch.object(checker, "EXECUTION_SOURCES", empty):
                    with self.assertRaisesRegex(AssertionError, "source digest: " + evolved[0]):
                        checker.check(record, replay_rng=False)
                # A snapshot that is not the recorded bytes excuses nothing.
                forged = empty / evolved[0]
                forged.parent.mkdir(parents=True)
                forged.write_bytes((checker.EXECUTION_SOURCES / evolved[0]).read_bytes() + b"\n")
                with patch.object(checker, "EXECUTION_SOURCES", empty):
                    with self.assertRaisesRegex(AssertionError, "source digest: " + evolved[0]):
                        checker.evolved_sources(record)
            # A recorded digest nothing matches fails whatever is packaged.
            broken = deepcopy(record)
            broken["source_sha256"]["clifford_qc/fermion.py"] = "0" * 64
            with self.assertRaisesRegex(AssertionError, "source digest: clifford_qc/fermion.py"):
                checker.check(broken, replay_rng=False)

    def test_noise_training_energy_vectors_must_match_candidate_count(self):
        record = json.loads(gzip.decompress(check_noise_aware_selection.RESULT.read_bytes()))
        original = record["outcomes"][0]["rows"][0]["training_energies"]
        for replacement in ([], original[:-1], original + [0.0]):
            with self.subTest(length=len(replacement)):
                broken = deepcopy(record)
                broken["outcomes"][0]["rows"][0]["training_energies"] = replacement
                with self.assertRaisesRegex(AssertionError, "training energy count"):
                    check_noise_aware_selection.check(broken, replay_rng=False)
                with self.assertRaisesRegex(AssertionError, "no invalid candidate pencils"):
                    check_noise_aware_selection.check_report(broken)

    def test_history_binds_sources_and_earliest_published_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.run(["git", *args], cwd=root, check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()
            def write(path, data):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            def commit(message):
                git("add", "-A")
                git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-q", "-m", message)
                return git("rev-parse", "HEAD")
            git("init", "-q")
            design_path = "docs/restart/noise_aware_selection_declaration.json"
            result_path = "docs/restart/noise_aware_selection_results.json.gz"
            producer = "benchmarks/run_noise_aware_selection.py"
            helper = "benchmarks/helper.py"
            design = b'{"fixture":true}\n'
            write(design_path, design)
            declaration = commit("declare")
            write(producer, b"# producer\n")
            write(helper, b"# helper\n")
            implementation = commit("implement")
            sources = (design_path, producer, helper)
            record = {"source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                        for name in sources}, "observation": 1.0}
            def write_result(value):
                write(result_path, gzip.compress(json.dumps(value, sort_keys=True).encode(), mtime=0))
            write_result(record)
            commit("publish")
            with patch.multiple(check_noise_aware_selection, ROOT=root, RESULT=root/result_path,
                    DECLARATION_COMMIT=declaration, IMPLEMENTATION_COMMIT=implementation,
                    DECLARATION_SHA256=hashlib.sha256(design).hexdigest(),
                    BOUND_SOURCES=frozenset(sources)):
                check_noise_aware_selection.verify_history(record)
                changed_source = deepcopy(record)
                write(helper, b"# changed helper\n")
                changed_source["source_sha256"][helper] = hashlib.sha256((root/helper).read_bytes()).hexdigest()
                with self.assertRaisesRegex(AssertionError, "implementation source differs"):
                    check_noise_aware_selection.verify_history(changed_source)
                write(helper, b"# helper\n")
                changed_result = deepcopy(record)
                changed_result["observation"] = 2.0
                with self.assertRaisesRegex(AssertionError, "published result payload differs"):
                    check_noise_aware_selection.verify_history(changed_result)
                git("rm", "-q", result_path)
                commit("remove result")
                write_result(changed_result)
                commit("re-add edited result")
                with self.assertRaisesRegex(AssertionError, "published result payload differs"):
                    check_noise_aware_selection.verify_history(changed_result)


if __name__ == "__main__":
    unittest.main()
