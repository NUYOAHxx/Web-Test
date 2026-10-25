"""Offline numerical/logic tests; not a ROS/RMW or robot integration test."""
import ast
import json
import math
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

import ik_state_observer_jazzy as m


def link(name, mass=None, xyz="0 0 0", geometry=False):
    inertial = "" if mass is None else f'<inertial><origin xyz="{xyz}"/><mass value="{mass}"/></inertial>'
    visual = "<visual/>" if geometry else ""
    return f'<link name="{name}">{inertial}{visual}</link>'


def joint(name, parent, child, kind="revolute", xyz="0 0 0", axis="0 1 0",
          extra="", limit='<limit lower="-3" upper="3" effort="100"/>'):
    return (f'<joint name="{name}" type="{kind}"><parent link="{parent}"/>'
            f'<child link="{child}"/><origin xyz="{xyz}"/><axis xyz="{axis}"/>'
            f'{limit}{extra}</joint>')


def robot(*parts):
    return '<robot name="test">'+"".join(parts)+"</robot>"


def one_arm(kind="revolute", mass=2, axis="0 1 0", extra="", limit=None):
    kwargs = {} if limit is None else {"limit": limit}
    return robot(link("world"), link("arm", mass, "1 0 0", True),
                 joint("j", "world", "arm", kind=kind, axis=axis, extra=extra, **kwargs))


def serial_xml():
    return robot(link("world"), link("a", 1, ".5 0 0", True), link("b", 2, ".5 0 0", True),
                 joint("j1", "world", "a"), joint("j2", "a", "b", xyz="1 0 0"))


class GeometryTests(unittest.TestCase):
    def test_rpy_rigid_transform_inverse(self):
        T = m.origin(ET.fromstring('<origin xyz="1 2 3" rpy=".2 -.4 .9"/>'))
        np.testing.assert_allclose(T @ m.rigid_inverse(T), np.eye(4), atol=1e-12)

    def test_origin_defaults(self):
        np.testing.assert_allclose(m.origin(None), np.eye(4))

    def test_axis_rotation(self):
        R = m.axis_rotation(np.array([0., 1., 0.]), math.pi/2)
        np.testing.assert_allclose(R @ [1., 0., 0.], [0., 0., -1.], atol=1e-12)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)

    def test_quaternion_normalization(self):
        R = m.quaternion_matrix([0., 0., 2., 2.])
        np.testing.assert_allclose(R @ [1., 0., 0.], [0., 1., 0.], atol=1e-12)

    def test_invalid_quaternion(self):
        for q in ([0., 0., 0., 0.], [0., 0., 0., math.nan]):
            with self.assertRaises(ValueError):
                m.quaternion_matrix(q)

    def test_bad_vector(self):
        for value in ("1 2", "1 nan 3", "1 2 3 4"):
            with self.assertRaises(ValueError):
                m.vector(value, (0., 0., 0.))


class UrdfTests(unittest.TestCase):
    def test_no_local_file_required(self):
        model = m.UrdfModel(one_arm())
        self.assertEqual(model.root, "world")
        self.assertEqual(model.independent, ["j"])
        self.assertEqual(len(model.sha256), 64)

    def test_reject_doctype(self):
        with self.assertRaises(ValueError):
            m.UrdfModel('<!DOCTYPE robot []>'+one_arm())

    def test_reject_nonrobot(self):
        with self.assertRaises(ValueError):
            m.UrdfModel("<data/>")

    def test_duplicate_link(self):
        with self.assertRaises(ValueError):
            m.UrdfModel(robot(link("a"), link("a")))

    def test_unsupported_floating_joint(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            m.UrdfModel(one_arm(kind="floating"))

    def test_missing_link(self):
        with self.assertRaises(ValueError):
            m.UrdfModel(robot(link("world"), joint("j", "world", "missing")))

    def test_disconnected_root(self):
        with self.assertRaises(ValueError):
            m.UrdfModel(robot(link("a"), link("b")))

    def test_zero_axis(self):
        with self.assertRaises(ValueError):
            m.UrdfModel(one_arm(axis="0 0 0"))

    def test_negative_mass(self):
        with self.assertRaises(ValueError):
            m.UrdfModel(one_arm(mass=-1))

    def test_reversed_limits(self):
        with self.assertRaises(ValueError):
            m.UrdfModel(one_arm(limit='<limit lower="1" upper="-1"/>'))

    def test_axis_default_is_urdf_x(self):
        xml = one_arm().replace('<axis xyz="0 1 0"/>', "")
        np.testing.assert_allclose(m.UrdfModel(xml).joints["j"].axis, [1., 0., 0.])

    def test_continuous_never_invents_hard_stops(self):
        model = m.UrdfModel(one_arm(kind="continuous"))
        self.assertIsNone(model.joints["j"].lower)
        q, _, _ = model.kinematics({"j": 7*math.pi})
        self.assertEqual(q["j"], 7*math.pi)
        self.assertIsNone(m.joint_margin(model.joints["j"], 7*math.pi))

    def test_missing_position_limits_are_unknown(self):
        j = m.UrdfModel(one_arm(limit='<limit effort="100"/>')).joints["j"]
        self.assertIsNone(j.lower)
        self.assertIsNone(m.joint_margin(j, 0.))

    def test_missing_or_zero_effort_no_fake_denominator(self):
        for limits in ("", '<limit effort="0"/>', '<limit effort="nan"/>'):
            j = m.UrdfModel(one_arm(limit=limits)).joints["j"]
            self.assertIsNone(j.effort)
            self.assertIsNone(m.load_ratio(10., j.effort))

    def test_mimic_cycle(self):
        xml = robot(link("world"), link("a"), link("b"),
                    joint("j1", "world", "a", extra='<mimic joint="j2"/>'),
                    joint("j2", "a", "b", extra='<mimic joint="j1"/>'))
        with self.assertRaisesRegex(ValueError, "mimic"):
            m.UrdfModel(xml)

    def test_mimic_resolution(self):
        xml = serial_xml().replace('</joint></robot>',
                                   '<mimic joint="j1" multiplier="-2" offset=".1"/></joint></robot>')
        model = m.UrdfModel(xml)
        self.assertEqual(model.independent, ["j1"])
        q, _, _ = model.kinematics({"j1": .2})
        self.assertAlmostEqual(q["j2"], -.3)

    def test_out_of_range_positions_not_clipped(self):
        model = m.UrdfModel(one_arm())
        q, _, _ = model.kinematics({"j": 4.})
        self.assertEqual(q["j"], 4.)
        self.assertLess(m.joint_margin(model.joints["j"], q["j"]), 0.)


class PhysicsTests(unittest.TestCase):
    def test_single_joint_gravity_sign_and_com(self):
        result = m.UrdfModel(one_arm()).evaluate({"j": 0.}, [0., 0., -10.])
        self.assertAlmostEqual(result["mass_kg"], 2.)
        np.testing.assert_allclose(result["com"], [1., 0., 0.])
        self.assertAlmostEqual(result["gravity_actuated"]["j"], -20.)
        self.assertAlmostEqual(m.load_ratio(-20., 100.), .2)

    def test_rotation_changes_com_and_torque(self):
        result = m.UrdfModel(one_arm()).evaluate({"j": math.pi/2}, [0., 0., -10.])
        np.testing.assert_allclose(result["com"], [0., 0., -1.], atol=1e-12)
        self.assertAlmostEqual(result["gravity_actuated"]["j"], 0.)

    def test_serial_subtree_accumulation(self):
        result = m.UrdfModel(serial_xml()).evaluate({"j1": 0., "j2": 0.}, [0., 0., -10.])
        self.assertAlmostEqual(result["gravity_actuated"]["j1"], -35.)
        self.assertAlmostEqual(result["gravity_actuated"]["j2"], -10.)
        np.testing.assert_allclose(result["com"], [3.5/3., 0., 0.])

    def test_branching_subtree_order(self):
        xml = robot(link("world"), link("a", 1, "1 0 0"), link("b", 2, "0 1 0"),
                    link("c", 3, "1 0 0"), joint("ja", "world", "a"),
                    joint("jb", "world", "b"), joint("jc", "a", "c", xyz="1 0 0"))
        result = m.UrdfModel(xml).evaluate({"ja": 0., "jb": 0., "jc": 0.}, [0., 0., -10.])
        self.assertAlmostEqual(result["mass_kg"], 6.)
        np.testing.assert_allclose(result["com"], [7./6., 2./6., 0.])

    def test_prismatic_effort_unit_force(self):
        model = m.UrdfModel(one_arm(kind="prismatic", axis="0 0 1"))
        result = model.evaluate({"j": .2}, [0., 0., -10.])
        self.assertAlmostEqual(result["gravity_actuated"]["j"], 20.)

    def test_unknown_gravity_does_not_hide_com(self):
        result = m.UrdfModel(one_arm()).evaluate({"j": 0.}, None)
        self.assertIsNotNone(result["com"])
        self.assertEqual(result["gravity_actuated"], {})

    def test_missing_mass_default_suppresses_estimates(self):
        xml = robot(link("world"), link("a", 1, "1 0 0", True), link("b", geometry=True),
                    joint("j1", "world", "a"), joint("j2", "a", "b"))
        model = m.UrdfModel(xml)
        result = model.evaluate({"j1": 0., "j2": 0.}, [0., 0., -10.])
        self.assertIsNone(result["com"])
        self.assertEqual(result["gravity_actuated"], {})
        self.assertEqual(result["mass_state"], "PARTIAL_MODEL")
        partial = model.evaluate({"j1": 0., "j2": 0.}, [0., 0., -10.], True)
        self.assertIsNotNone(partial["com"])
        self.assertEqual(partial["mass_state"], "PARTIAL_MODEL")

    def test_all_zero_mass_not_fake_com(self):
        result = m.UrdfModel(one_arm(mass=0)).evaluate({"j": 0.}, [0., 0., -10.], True)
        self.assertIsNone(result["com"])
        self.assertEqual(result["mass_state"], "NO_POSITIVE_MASS")

    def test_gravity_matches_potential_gradient(self):
        model = m.UrdfModel(serial_xml())
        g, q, eps = np.array([1., -2., -9.]), {"j1": .3, "j2": -.7}, 1e-6
        result = model.evaluate(q, g)

        def potential(values):
            _, frames, _ = model.kinematics(values)
            return -sum(l.mass*float(g @ m.point_transform(frames[n], l.local_com))
                        for n, l in model.links.items())

        for name in q:
            qp, qm = dict(q), dict(q)
            qp[name] += eps
            qm[name] -= eps
            gradient = (potential(qp)-potential(qm))/(2*eps)
            self.assertAlmostEqual(result["gravity_actuated"][name], gradient, places=6)

    def test_mimic_effort_virtual_work(self):
        xml = serial_xml().replace('</joint></robot>',
                                   '<mimic joint="j1" multiplier="-.7" offset=".2"/></joint></robot>')
        model, g, eps = m.UrdfModel(xml), np.array([0., 0., -10.]), 1e-6
        q = .4

        def potential(value):
            _, frames, _ = model.kinematics({"j1": value})
            return -sum(l.mass*float(g @ m.point_transform(frames[n], l.local_com))
                        for n, l in model.links.items())

        result = model.evaluate({"j1": q}, g)
        self.assertAlmostEqual(result["gravity_actuated"]["j1"],
                               (potential(q+eps)-potential(q-eps))/(2*eps), places=6)


class StateCacheTests(unittest.TestCase):
    def test_name_mapping_not_array_order(self):
        c = m.JointCache()
        c.update(["b", "a"], [2., 1.], [], 10_000_000_000, 10.)
        s = c.snapshot(["a", "b"], 10.1, 2., .5, 10_100_000_000)
        self.assertTrue(s["valid"])
        self.assertEqual(s["positions"], {"a": 1., "b": 2.})
        self.assertEqual(s["efforts"], {})

    def test_missing_joint_not_filled_with_zero(self):
        c = m.JointCache()
        c.update(["a"], [1.], [], 0, 1.)
        self.assertFalse(c.snapshot(["a", "b"], 1.1, 2., .5)["valid"])

    def test_receipt_timeout(self):
        c = m.JointCache()
        c.update(["a"], [1.], [], 0, 1.)
        self.assertFalse(c.snapshot(["a"], 4., 2., .5)["valid"])

    def test_ros_stamp_timeout(self):
        c = m.JointCache()
        c.update(["a"], [1.], [], 1_000_000_000, 10.)
        self.assertFalse(c.snapshot(["a"], 10.1, 2., .5, 5_000_000_000)["valid"])

    def test_future_clock_mismatch(self):
        c = m.JointCache()
        c.update(["a"], [1.], [], 10_000_000_000, 1.)
        self.assertFalse(c.snapshot(["a"], 1.1, 2., .5, 1_000_000_000)["valid"])

    def test_skew_between_partial_updates(self):
        c = m.JointCache()
        c.update(["a"], [1.], [], 1_000_000_000, 1.)
        c.update(["b"], [2.], [], 2_000_000_000, 2.)
        self.assertFalse(c.snapshot(["a", "b"], 2.1, 2., .5, 2_100_000_000)["valid"])

    def test_out_of_order_does_not_refresh_position(self):
        c = m.JointCache()
        c.update(["a"], [2.], [], 2_000_000_000, 2.)
        c.update(["a"], [1.], [], 1_000_000_000, 3.)
        self.assertEqual(c.positions["a"].value, 2.)
        self.assertEqual(c.positions["a"].received, 2.)

    def test_duplicate_names_rejected(self):
        with self.assertRaises(ValueError):
            m.JointCache().update(["a", "a"], [1., 2.], [], 0, 0.)

    def test_array_lengths_rejected(self):
        with self.assertRaises(ValueError):
            m.JointCache().update(["a", "b"], [1.], [], 0, 0.)

    def test_nan_position_rejected_atomically(self):
        c = m.JointCache()
        with self.assertRaises(ValueError):
            c.update(["a", "b"], [1., math.nan], [], 0, 0.)
        self.assertEqual(c.positions, {})

    def test_effort_only_does_not_refresh_q(self):
        c = m.JointCache()
        c.update(["a"], [1.], [], 0, 0.)
        c.update(["a"], [], [2.], 0, 3.)
        s = c.snapshot(["a"], 3., 2., .5)
        self.assertFalse(s["valid"])
        self.assertEqual(s["efforts"]["a"], 2.)

    def test_nan_and_stale_effort_are_unknown(self):
        c = m.JointCache()
        c.update(["a"], [1.], [2.], 0, 0.)
        c.update(["a"], [1.], [math.nan], 0, .1)
        self.assertEqual(c.snapshot(["a"], .2, 2., .5)["efforts"], {})

    def test_allowed_name_filter(self):
        c = m.JointCache()
        c.update(["a", "unknown"], [1., 5.], [], 0, 0., {"a"})
        self.assertEqual(set(c.positions), {"a"})

    def test_zero_stamp_is_explicit_and_uses_latest_tf_policy(self):
        c = m.JointCache()
        c.update(["a", "b"], [1., 2.], [], 0, 1.)
        s = c.snapshot(["a", "b"], 1.1, 2., .5)
        self.assertTrue(s["valid"])
        self.assertEqual(s["stamp_ns"], 0)
        self.assertEqual(s["unstamped"], ["a", "b"])


class InterfaceTests(unittest.TestCase):
    def test_support_projection(self):
        d = m.support_projection(np.array([0., 0., 1.]), np.array([1., 0., -10.]),
                                 np.eye(4), [.2, .3])
        np.testing.assert_allclose(d["projection"], [.1, 0., 0.])
        self.assertAlmostEqual(d["margin_m"], .1)

    def test_support_unavailable_or_invalid(self):
        self.assertEqual(m.support_projection(None, None, None)["state"], "UNAVAILABLE")
        d = m.support_projection(np.array([0., 0., 1.]), np.array([0., 0., 10.]), np.eye(4))
        self.assertEqual(d["state"], "INVALID_PLANE_OR_GRAVITY")

    def test_reference_does_not_override(self):
        model = m.UrdfModel(one_arm())
        old = model.joints["j"].lower
        notes = m.reference_notes(model, {"joints": {"j": {"lower_deg": -90, "upper_deg": 90}}})
        self.assertTrue(notes)
        self.assertEqual(model.joints["j"].lower, old)

    def test_continuous_reference_is_not_hard_stop(self):
        model = m.UrdfModel(one_arm(kind="continuous"))
        notes = m.reference_notes(model, {"joints": {"j": {"lower_deg": -360, "upper_deg": 360}}})
        self.assertIn("NOT a hard stop", notes[0])
        self.assertIsNone(model.joints["j"].lower)

    def test_position_limit_text_distinguishes_continuous(self):
        j = m.UrdfModel(one_arm(kind="continuous")).joints["j"]
        self.assertIn("no hard position stop", m.position_bounds_text(j))

    def test_position_limit_text_prismatic_uses_metres(self):
        j = m.UrdfModel(one_arm(kind="prismatic")).joints["j"]
        self.assertTrue(m.position_bounds_text(j).endswith(" m"))

    def test_ros_args_split(self):
        app, ros = m.split_ros_args(["--rate", "5", "--ros-args", "-p", "use_sim_time:=true"])
        self.assertEqual(app, ["--rate", "5"])
        self.assertEqual(ros[0], "--ros-args")

    def test_no_urdf_required_cli(self):
        args = m.argument_parser().parse_args([])
        m.validate_args(args)
        self.assertEqual(args.joint_topic, "/joint_states")

    def test_invalid_support_config(self):
        args = m.argument_parser().parse_args(["--support-half-extents", ".1", ".2"])
        with self.assertRaises(ValueError):
            m.validate_args(args)

    def test_python_312_syntax(self):
        ast.parse(Path(m.__file__).read_text(), feature_version=(3, 12))

    def test_only_marker_publisher_no_ik_or_tf_output(self):
        tree = ast.parse(Path(m.__file__).read_text())
        publishers = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                      isinstance(n.func, ast.Attribute) and n.func.attr == "create_publisher"]
        self.assertEqual(len(publishers), 1)
        self.assertIsInstance(publishers[0].args[0], ast.Name)
        self.assertEqual(publishers[0].args[0].id, "MarkerArray")
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any("ha_arm_ik" in name or "action" in name for name in imports))

    def test_log_no_overwrite_and_standard_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.csv"
            log = m.CsvLog(path, {"test": math.nan})
            log.write(1, "sha", "test", {"n": np.array([1., math.nan])})
            log.close()
            self.assertEqual(json.loads(Path(str(path)+".meta.json").read_text()), {"test": None})
            with self.assertRaises(FileExistsError):
                m.CsvLog(path, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)