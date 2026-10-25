"""Mocked node workflow tests. These DO NOT test actual rclpy/DDS/TF transport."""
import contextlib
import io
import sys
import types
import unittest
from unittest.mock import patch

import ik_state_observer_jazzy as m
from test_observer import one_arm, serial_xml


def ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


class Time:
    def __init__(self, nanoseconds=0, clock_type=None, **kwargs):
        self.nanoseconds = nanoseconds

    def to_msg(self):
        return ns(sec=self.nanoseconds//1_000_000_000, nanosec=self.nanoseconds % 1_000_000_000)


class Duration:
    def __init__(self, seconds=0., **kwargs):
        self.seconds = seconds


class Point:
    def __init__(self):
        self.x = self.y = self.z = 0.


class Marker:
    ADD, DELETE, SPHERE, TEXT_VIEW_FACING, LINE_STRIP = 0, 2, 2, 9, 4

    def __init__(self):
        self.header = ns(frame_id="", stamp=None)
        self.pose = ns(position=Point(), orientation=ns(x=0., y=0., z=0., w=0.))
        self.color = ns(r=0., g=0., b=0., a=0.)
        self.scale = Point()
        self.lifetime = ns(sec=0, nanosec=0)
        self.ns, self.id, self.type, self.action = "", 0, 0, 0
        self.text, self.points = "", []


class MarkerArray:
    def __init__(self):
        self.markers = []


class JointState:
    def __init__(self, names=("j",), positions=(0.,), effort=(), stamp=1_000_000_000):
        self.header = ns(stamp=Time(stamp).to_msg())
        self.name, self.position, self.effort = list(names), list(positions), list(effort)


class GetParameters:
    class Request:
        def __init__(self):
            self.names = []


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(("info", msg))

    def warning(self, msg):
        self.messages.append(("warning", msg))

    def error(self, msg):
        self.messages.append(("error", msg))


def modules_for(driver, model_xml=None):
    registry, environment = {}, ns(node=None, alive=False, clock_ns=1_000_000_000,
                                  model_xml=model_xml, buffers=[])

    def module(name, **attributes):
        obj = types.ModuleType(name)
        obj.__dict__.update(attributes)
        registry[name] = obj
        return obj

    class FakeFuture:
        def done(self):
            return True

        def result(self):
            return ns(values=[ns(type=4, string_value=environment.model_xml)])

        def add_done_callback(self, callback):
            callback(self)

    class Client:
        def __init__(self):
            self.requests = []

        def service_is_ready(self):
            return environment.model_xml is not None

        def call_async(self, request):
            self.requests.append(request.names)
            return FakeFuture()

    class Publisher:
        def __init__(self, kind, topic):
            self.kind, self.topic, self.messages = kind, topic, []

        def publish(self, msg):
            self.messages.append(msg)

    class Node:
        def __init__(self, name):
            self.logger, self.subscriptions, self.publishers = FakeLogger(), [], []
            self.client, self.destroyed = None, False
            environment.node = self

        def create_subscription(self, kind, topic, callback, qos):
            self.subscriptions.append((kind, topic, callback, qos))
            return object()

        def create_publisher(self, kind, topic, qos):
            pub = Publisher(kind, topic)
            self.publishers.append(pub)
            return pub

        def create_client(self, kind, service):
            self.client = Client()
            self.service = service
            return self.client

        def create_timer(self, period, callback):
            return ns(period=period, callback=callback)

        def get_clock(self):
            return ns(now=lambda: Time(environment.clock_ns))

        def get_logger(self):
            return self.logger

        def destroy_node(self):
            self.destroyed = True

    class Buffer:
        def __init__(self, **kwargs):
            self.fail = False
            environment.buffers.append(self)

        def lookup_transform(self, target, source, when, **kwargs):
            if self.fail:
                raise RuntimeError("mock TF unavailable")
            return ns(transform=ns(translation=ns(x=0., y=0., z=0.),
                                   rotation=ns(x=0., y=0., z=0., w=1.)))

    def initialize(**kwargs):
        environment.alive = True

    def shutdown():
        environment.alive = False

    module("rclpy", init=initialize, shutdown=shutdown, ok=lambda: environment.alive,
           spin=lambda node: driver(node, environment))
    module("rclpy.node", Node=Node)
    module("rclpy.time", Time=Time)
    module("rclpy.duration", Duration=Duration)
    module("rclpy.clock", ClockType=ns(ROS_TIME=1))
    module("rclpy.qos", QoSProfile=lambda **kw: ns(**kw),
           ReliabilityPolicy=ns(RELIABLE=1), DurabilityPolicy=ns(TRANSIENT_LOCAL=1),
           qos_profile_sensor_data=ns(best_effort=True))
    module("rclpy.executors", ExternalShutdownException=type("ExternalShutdownException", (Exception,), {}))
    for parent in ("rcl_interfaces", "sensor_msgs", "std_msgs", "visualization_msgs", "geometry_msgs"):
        module(parent)
    module("rcl_interfaces.srv", GetParameters=GetParameters)
    module("sensor_msgs.msg", JointState=JointState)
    module("std_msgs.msg", String=type("String", (), {}))
    module("visualization_msgs.msg", Marker=Marker, MarkerArray=MarkerArray)
    module("geometry_msgs.msg", Point=Point)
    module("tf2_ros", Buffer=Buffer, TransformListener=lambda *a, **kw: object())
    return registry, environment


def active_names(node):
    return {msg.ns for msg in node.publishers[0].messages[-1].markers if msg.action == Marker.ADD}


class MockNodeTests(unittest.TestCase):
    def run_node(self, driver, arguments=(), model_xml=None):
        args = m.argument_parser().parse_args(arguments)
        modules, environment = modules_for(driver, model_xml)
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            m.run_ros(args, [])
        self.assertTrue(environment.node.destroyed)
        return environment

    def ready(self, node):
        node.description_cb(ns(data=one_arm()))
        node.joint_cb(JointState())
        node.tick()

    def test_description_topic_to_markers_no_extra_publishers(self):
        def driver(node, env):
            self.ready(node)
            self.assertIn("ik_observer/com", active_names(node))
            self.assertTrue(node.last_status.startswith("OK"))
            self.assertEqual(len(node.publishers), 1)
            self.assertIs(node.publishers[0].kind, MarkerArray)
            self.assertEqual(node.publishers[0].topic, "/ha_ik_monitor/markers")
            self.assertEqual(len(node.subscriptions), 2)
        self.run_node(driver)

    def test_getparameters_fallback(self):
        def driver(node, env):
            node.tick()
            self.assertIsNotNone(node.model)
            self.assertEqual(node.client.requests, [["robot_description"]])
            self.assertEqual(node.service, "/robot_state_publisher/get_parameters")
            node.joint_cb(JointState())
            node.tick()
            self.assertIn("ik_observer/com", active_names(node))
        self.run_node(driver, model_xml=one_arm())

    def test_waiting_model_does_not_invent_com(self):
        def driver(node, env):
            node.tick()
            self.assertIn("WAIT_MODEL", node.last_status)
            self.assertNotIn("ik_observer/com", active_names(node))
        self.run_node(driver)

    def test_stale_state_deletes_previous_markers(self):
        def driver(node, env):
            self.ready(node)
            node.cache.positions["j"].received -= 3.
            node.tick()
            self.assertIn("WAIT_STATE", node.last_status)
            messages = node.publishers[0].messages[-1].markers
            self.assertTrue(any(x.ns == "ik_observer/com" and x.action == Marker.DELETE for x in messages))
        self.run_node(driver)

    def test_unknown_gravity_tf_keeps_com_but_no_load_estimate(self):
        def driver(node, env):
            env.buffers[0].fail = True
            self.ready(node)
            self.assertIn("GRAVITY_TF_UNKNOWN", node.last_status)
            self.assertIn("ik_observer/com", active_names(node))
        self.run_node(driver, ["--frame-prefix", "robot/"])

    def test_support_rectangle_opt_in(self):
        def driver(node, env):
            self.ready(node)
            self.assertIn("ik_observer/support_rectangle", active_names(node))
        self.run_node(driver, ["--support-frame", "world", "--support-half-extents", "2", "2"])

    def test_default_does_not_draw_assumed_support(self):
        def driver(node, env):
            self.ready(node)
            self.assertNotIn("ik_observer/support_rectangle", active_names(node))
        self.run_node(driver)

    def test_joint_labels_and_reference_state_not_clipped(self):
        def driver(node, env):
            node.description_cb(ns(data=one_arm()))
            node.joint_cb(JointState(positions=[4.]))
            node.tick()
            self.assertEqual(node.cache.positions["j"].value, 4.)
            self.assertIn("URDF_LIMIT_VIOLATION", node.last_status)
            self.assertIn("ik_observer/joint_labels", active_names(node))
        self.run_node(driver, ["--joint-labels"])

    def test_model_reload_requires_new_state(self):
        def driver(node, env):
            self.ready(node)
            node.description_cb(ns(data=one_arm(mass=3)))
            node.tick()
            self.assertIn("WAIT_STATE", node.last_status)
            self.assertNotIn("ik_observer/com", active_names(node))
        self.run_node(driver)

    def test_bad_model_update_disables_old_estimates(self):
        def driver(node, env):
            self.ready(node)
            node.description_cb(ns(data="<bad/>"))
            node.tick()
            self.assertIsNone(node.model)
            self.assertNotIn("ik_observer/com", active_names(node))
        self.run_node(driver)

    def test_clock_rewind_clears_state(self):
        def driver(node, env):
            self.ready(node)
            env.clock_ns = 500_000_000
            node.tick()
            self.assertIn("WAIT_STATE", node.last_status)
            self.assertEqual(node.cache.positions, {})
        self.run_node(driver)

    def test_shutdown_context_does_not_publish_after_shutdown(self):
        def driver(node, env):
            self.ready(node)
            env.alive = False
        self.run_node(driver)


if __name__ == "__main__":
    unittest.main(verbosity=2)