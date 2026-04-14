"""Tests for demo scenario and catalog helpers."""

from simulation.demo_scenarios import (
    due_scenario_events,
    list_demo_models,
    quota_preset_to_resource_map,
    serialize_catalog,
)


class TestQuotaPresets:
    def test_quota_preset_returns_orca_shape_c_resource_map(self):
        resource_map = quota_preset_to_resource_map("aws_mixed_demo")
        assert set(resource_map.keys()) == {"instances", "quotas"}
        assert resource_map["instances"]
        assert resource_map["quotas"]
        assert {
            "instance_type",
            "gpu_type",
            "gpus_per_instance",
            "gpu_memory_gb",
            "vcpus",
            "quota_family",
            "cost_per_instance_hour_usd",
        } <= set(resource_map["instances"][0].keys())


class TestScenarios:
    def test_hero_elastic_has_pressure_rise_and_relief(self):
        due = due_scenario_events("hero_elastic", elapsed_seconds=120)
        event_ids = {event.event_id for event in due}
        assert "hero-pressure-rise" in event_ids
        assert "hero-pressure-relief" in event_ids

    def test_due_events_excludes_completed_ids(self):
        due = due_scenario_events(
            "kill_and_recover",
            elapsed_seconds=60,
            completed_event_ids={"kill-primary"},
        )
        assert due == []


class TestCatalog:
    def test_catalog_contains_models_presets_and_scenarios(self):
        catalog = serialize_catalog()
        assert {"models", "quota_presets", "scenarios"} <= set(catalog.keys())
        assert catalog["models"]
        assert catalog["quota_presets"]
        assert catalog["scenarios"]

    def test_demo_models_include_known_registry_entries(self):
        names = {choice.model_name for choice in list_demo_models()}
        assert "Qwen/Qwen3-32B" in names
