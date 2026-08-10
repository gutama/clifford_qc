import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmarks import run_prd_case_paper_suite as suite


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "benchmarks" / "configs" / "prd_case_paper_suite.json"


def manifest():
    return suite.load_manifest(CONFIG)


class TestPRDCasePaperSuite(unittest.TestCase):

    def test_manifest_preregisters_twelve_physical_instances_and_one_validation(self):
        document = manifest()
        primary = [system for system in document["systems"]
                   if system["analysis_role"] == "primary"]
        validation = [system for system in document["systems"]
                      if system["analysis_role"] == "construction_validation"]
        self.assertEqual(len(primary), 12)
        self.assertEqual(len(validation), 1)
        self.assertEqual(validation[0]["id"], "fcidump_h4_equilibrium")
        self.assertIs(validation[0]["count_as_distinct"], False)

    def test_default_plan_has_sixty_primary_tasks_and_one_validation_task(self):
        tasks = suite.build_tasks(manifest())
        self.assertEqual(len(tasks), 61)
        self.assertEqual(sum(task["analysis_role"] == "primary" for task in tasks),
                         60)
        self.assertEqual(sum(task["analysis_role"] == "construction_validation"
                             for task in tasks), 1)
        self.assertFalse(any(task["system"] == "hubbard_2x4_u4"
                             for task in tasks))

    def test_exploratory_scaling_is_opt_in(self):
        tasks = suite.build_tasks(manifest(), include_exploratory=True)
        scaling = [task for task in tasks if task["system"] == "hubbard_2x4_u4"]
        self.assertEqual([task["M"] for task in scaling], [3, 5, 7])

    def test_system_filter_is_exact_and_preserves_declared_M_grid(self):
        tasks = suite.build_tasks(manifest(), system_ids=("hubbard_2x3_u8",))
        self.assertEqual([task["M"] for task in tasks], [3, 5, 7, 11, 21])
        self.assertEqual({task["system"] for task in tasks}, {"hubbard_2x3_u8"})

    def test_manifest_hash_is_order_independent_but_content_sensitive(self):
        document = manifest()
        reordered = json.loads(json.dumps(document, sort_keys=True))
        self.assertEqual(suite.config_sha256(document),
                         suite.config_sha256(reordered))
        reordered["exact_simulation"]["word_budget_ratio"] = 3.0
        self.assertNotEqual(suite.config_sha256(document),
                            suite.config_sha256(reordered))

    def test_checkpoint_refuses_a_changed_preregistration(self):
        document = manifest()
        tasks = suite.build_tasks(document)[:1]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.json"
            checkpoint.write_text(
                json.dumps(suite._new_checkpoint(document, tasks)),
                encoding="utf-8")
            changed = json.loads(json.dumps(document))
            changed["exact_simulation"]["word_budget_ratio"] = 3.0
            with self.assertRaisesRegex(ValueError, "different manifest"):
                suite.load_checkpoint(checkpoint, changed, tasks)

    def test_claim_boundary_forbids_quantum_advantage_upgrade(self):
        boundaries = " ".join(manifest()["claim_boundaries"]).lower()
        self.assertIn("no quantum-advantage", boundaries)
        self.assertIn("not an implementable hardware primitive", boundaries)

    def test_acase_baselines_are_required_but_not_mislabelled_as_produced(self):
        document = manifest()
        stages = {method["id"]: method["execution_stage"]
                  for method in document["methods"]}
        self.assertEqual(stages["acase_determinant"], "matched_acase")
        self.assertEqual(stages["acase_word"], "matched_acase")
        self.assertEqual(stages["davidson"], "preconditioned_expansion")

    def test_validation_task_keeps_its_target_and_tolerance(self):
        tasks = suite.build_tasks(manifest(),
                                  system_ids=("fcidump_h4_equilibrium",))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["validation_target"], "h4_equilibrium")
        self.assertEqual(tasks[0]["energy_tolerance"], 1e-10)

    def test_unanchored_word_pricing_has_a_hard_abort(self):
        exact = manifest()["exact_simulation"]
        self.assertGreaterEqual(exact["absolute_packet_word_abort"],
                                exact["grouping_word_limit"])
        extension = exact["accuracy_only_packet_extension"]
        self.assertIs(extension["implemented_by_this_driver"], False)

    def test_runtime_guard_caps_unanchored_and_oversized_pricing(self):
        calls = []

        def original_pricer(*args, abort_above=None, **kwargs):
            calls.append(abort_above)
            return abort_above

        original_builder = object()
        fake = SimpleNamespace(
            phase10=SimpleNamespace(build_system=original_builder),
            PRIMARY_SYSTEMS=("old",),
            price_packet_basis=original_pricer,
        )
        with suite._registered_for_one_run(
                fake, "new", {"kind": "hubbard"}, absolute_word_abort=20000):
            self.assertEqual(fake.price_packet_basis(abort_above=None), 20000)
            self.assertEqual(fake.price_packet_basis(abort_above=50000), 20000)
            self.assertEqual(fake.price_packet_basis(abort_above=1000), 1000)
            self.assertEqual(fake.PRIMARY_SYSTEMS, ("new",))
        self.assertEqual(calls, [20000, 20000, 1000])
        self.assertIs(fake.phase10.build_system, original_builder)
        self.assertIs(fake.price_packet_basis, original_pricer)

    def test_phase10_registry_uses_the_unpatched_builder(self):
        calls = []

        def original_builder(name):
            calls.append(name)
            return "model", {"source": "control"}

        fake = SimpleNamespace(
            phase10=SimpleNamespace(build_system=original_builder),
            PRIMARY_SYSTEMS=("old",),
            price_packet_basis=lambda *args, **kwargs: None,
        )
        builder = {"kind": "phase10_registry",
                   "source_key": "fcidump_h4_equilibrium"}
        suite._MODEL_CACHE.clear()
        with suite._registered_for_one_run(
                fake, "validation", builder, absolute_word_abort=20000):
            model, construction = fake.phase10.build_system("validation")
        self.assertEqual(model, "model")
        self.assertEqual(construction, {"source": "control"})
        self.assertEqual(calls, ["fcidump_h4_equilibrium"])


if __name__ == "__main__":
    unittest.main()
