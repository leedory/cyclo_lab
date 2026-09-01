"""Round-trip tests for SG2 HDF5 action representation conversion."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[3]
ISAACLAB_SOURCE = REPO / "third_party" / "IsaacLab" / "source" / "isaaclab"
CYCLO_LAB_SOURCE = REPO / "source" / "cyclo_lab"
for source_root in (ISAACLAB_SOURCE, CYCLO_LAB_SOURCE):
    sys.path.insert(0, str(source_root))

from cyclo_lab.robot_specs.ffw.sg2 import (  # noqa: E402
    FFW_SG2_MOBILE_ACTION_NAMES,
    FFW_SG2_SDG_ACTION_NAMES,
)


def _load_converter():
    path = (
        REPO
        / "scripts"
        / "sim2real"
        / "imitation_learning"
        / "mimic"
        / "action_data_converter.py"
    )
    spec = importlib.util.spec_from_file_location("sg2_action_data_converter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _decode_json_attr(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value) if isinstance(value, str) else value


def _write_joint22_source(path: Path) -> tuple[np.ndarray, np.ndarray]:
    actions = np.arange(66, dtype=np.float32).reshape(3, 22) / 100.0
    joint_targets = actions[:, :19] + 0.25
    left_eef = np.zeros((3, 7), dtype=np.float32)
    right_eef = np.zeros((3, 7), dtype=np.float32)
    left_eef[:, 3] = 1.0
    right_eef[:, 3] = 1.0

    with h5py.File(path, "w") as hdf:
        data = hdf.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "Task525Test", "type": 2})
        data.attrs["total"] = len(actions)
        data.attrs["robot_contract_id"] = "ffw_sg2_rev1_mobile_22d_v1"
        data.attrs["action_names"] = json.dumps(list(FFW_SG2_MOBILE_ACTION_NAMES))
        data.attrs["action_units"] = json.dumps(["unit"] * 22)
        data.attrs["action_semantics"] = (
            "pre_step_joint_position_19_plus_body_velocity_3"
        )
        data.attrs["eef_action_frame"] = "none"

        demo = data.create_group("demo_0")
        demo.attrs["success"] = True
        demo.attrs["num_samples"] = len(actions)
        demo.create_dataset("actions", data=actions)
        obs = demo.create_group("obs")
        obs.create_dataset("left_eef_pose", data=left_eef)
        obs.create_dataset("right_eef_pose", data=right_eef)
        obs.create_dataset("joint_pos_target", data=joint_targets)
    return actions, joint_targets


def test_ffw_sg2_joint22_ik_round_trip_rewrites_contract(tmp_path):
    converter = _load_converter()
    source = tmp_path / "joint22.hdf5"
    ik = tmp_path / "eef22.hdf5"
    joint = tmp_path / "joint22_round_trip.hdf5"
    source_actions, joint_targets = _write_joint22_source(source)

    converter.process_dataset(str(source), str(ik), "ik", "FFW_SG2")
    with h5py.File(ik, "r") as hdf:
        data = hdf["data"]
        output_actions = np.asarray(data["demo_0/actions"])
        assert data.attrs["robot_contract_id"] == (
            "ffw_sg2_task525_locomanipulation_sdg_eef22_v1"
        )
        assert data.attrs["source_robot_contract_id"] == (
            "ffw_sg2_rev1_mobile_22d_v1"
        )
        assert data.attrs["action_representation"] == "ik"
        assert data.attrs["eef_action_frame"] == "robot_root"
        assert data.attrs["mimic_trajectory_source"] == "achieved_eef_pose"
        assert tuple(_decode_json_attr(data.attrs["action_names"])) == tuple(
            FFW_SG2_SDG_ACTION_NAMES
        )
        np.testing.assert_allclose(output_actions[:, 19:22], source_actions[:, 19:22])

    converter.process_dataset(str(ik), str(joint), "joint", "FFW_SG2")
    with h5py.File(joint, "r") as hdf:
        data = hdf["data"]
        output_actions = np.asarray(data["demo_0/actions"])
        assert data.attrs["robot_contract_id"] == "ffw_sg2_rev1_mobile_22d_v1"
        assert data.attrs["source_robot_contract_id"] == (
            "ffw_sg2_task525_locomanipulation_sdg_eef22_v1"
        )
        assert data.attrs["action_representation"] == "joint"
        assert data.attrs["eef_action_frame"] == "none"
        assert data.attrs["mimic_trajectory_source"] == "none"
        assert tuple(_decode_json_attr(data.attrs["action_names"])) == tuple(
            FFW_SG2_MOBILE_ACTION_NAMES
        )
        np.testing.assert_allclose(output_actions[:, :19], joint_targets)
        np.testing.assert_allclose(output_actions[:, 19:22], source_actions[:, 19:22])


def test_lerobot_export_rejects_all_explicit_eef_representations():
    source = (
        REPO
        / "scripts"
        / "sim2real"
        / "imitation_learning"
        / "data_converter"
        / "isaaclab2lerobot.py"
    ).read_text(encoding="utf-8")
    assert '"eef" in normalized_contract_id' in source
    assert 'normalized_representation in {"ik", "eef"}' in source
