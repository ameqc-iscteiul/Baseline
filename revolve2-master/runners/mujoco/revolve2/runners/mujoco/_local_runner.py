import concurrent.futures
import math
import os
import tempfile
from typing import List, Optional, Union, Callable

import cv2
import mujoco
import mujoco_viewer
import numpy as np
import numpy.typing as npt

try:
    import logging

    old_len = len(logging.root.handlers)

    from dm_control import mjcf

    new_len = len(logging.root.handlers)

    assert (
        old_len + 1 == new_len
    ), "dm_control not adding logging handler as expected. Maybe they fixed their annoying behaviour? https://github.com/deepmind/dm_control/issues/314https://github.com/deepmind/dm_control/issues/314"

    logging.root.removeHandler(logging.root.handlers[-1])
except Exception as e:
    print("Failed to fix absl logging bug", e)
    pass

from pyrr import Quaternion, Vector3
from revolve2.core.physics.actor.urdf import to_urdf as physbot_to_urdf
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    BatchResults,
    Environment,
    EnvironmentResults,
    EnvironmentState,
    RecordSettings,
    Runner,
)


class LocalRunner(Runner):
    """Runner for simulating using Mujoco."""

    _headless: bool
    _start_paused: bool
    _num_simulators: int

    def __init__(
        self,
        headless: bool = False,
        start_paused: bool = False,
        num_simulators: int = 1,
    ):
        """
        Initialize this object.

        :param headless: If True, the simulation will not be rendered. This drastically improves performance.
        :param start_paused: If True, start the simulation paused. Only possible when not in headless mode.
        :param num_simulators: The number of simulators to deploy in parallel. They will take one core each but will share space on the main python thread for calculating control.
        """
        assert (
            headless or num_simulators == 1
        ), "Cannot have parallel simulators when visualizing."

        assert not (
            headless and start_paused
        ), "Cannot start simulation paused in headless mode."

        self._headless = headless
        self._start_paused = start_paused
        self._num_simulators = num_simulators

    @classmethod
    def _run_environment(
        cls,
        env_index: int,
        env_descr: Environment,
        headless: bool,
        record_settings: Optional[RecordSettings],
        start_paused: bool,
        control_step: float,
        sample_step: float,
        simulation_time: int,
    ) -> EnvironmentResults:
        logging.info(f"Environment {env_index}")

        model = mujoco.MjModel.from_xml_string(cls._make_mjcf(env_descr))

        # TODO initial dof state
        data = mujoco.MjData(model)

        initial_targets = [
            dof_state
            for posed_actor in env_descr.actors
            for dof_state in posed_actor.dof_states
        ]
        cls._set_dof_targets(data, initial_targets)

        for posed_actor in env_descr.actors:
            posed_actor.dof_states

        if not headless or record_settings is not None:
            viewer = mujoco_viewer.MujocoViewer(
                model,
                data,
            )
            viewer._render_every_frame = False  # Private but functionality is not exposed and for now it breaks nothing.
            viewer._paused = start_paused

        if record_settings is not None:
            video_step = 1 / record_settings.fps
            video_file_path = f"{record_settings.video_directory}/{env_index}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                video_file_path,
                fourcc,
                record_settings.fps,
                (viewer.viewport.width, viewer.viewport.height),
            )

            viewer._hide_menu = True

        last_control_time = 0.0
        last_sample_time = 0.0
        last_video_time = 0.0  # time at which last video frame was saved

        results = EnvironmentResults([])

        # sample initial state
        results.environment_states.append(
            EnvironmentState(0.0, cls._get_actor_states(env_descr, data, model))
        )

        while (time := data.time) < simulation_time:

            # do control if it is time
            if time >= last_control_time + control_step:
                last_control_time = math.floor(time / control_step) * control_step
                control_user = ActorControl()
                env_descr.controller.control(control_step, control_user)
                actor_targets = control_user._dof_targets
                actor_targets.sort(key=lambda t: t[0])
                targets = [
                    target
                    for actor_target in actor_targets
                    for target in actor_target[1]
                ]
                cls._set_dof_targets(data, targets)

            # sample state if it is time
            if time >= last_sample_time + sample_step:
                last_sample_time = int(time / sample_step) * sample_step
                results.environment_states.append(
                    EnvironmentState(
                        time, cls._get_actor_states(env_descr, data, model)
                    )
                )

            # step simulation
            mujoco.mj_step(model, data)

            # render if not headless. also render when recording and if it time for a new video frame.
            if not headless or (
                record_settings is not None and time >= last_video_time + video_step
            ):
                viewer.render()

            # capture video frame if it's time
            if record_settings is not None and time >= last_video_time + video_step:
                last_video_time = int(time / video_step) * video_step

                img = viewer.read_pixels()
                video.write(img)

        if not headless or record_settings is not None:
            viewer.close()

        if record_settings is not None:
            video.release()

        # sample one final time
        results.environment_states.append(
            EnvironmentState(time, cls._get_actor_states(env_descr, data, model))
        )

        return results

    async def run_batch(
        self, batch: Batch, record_settings: Optional[RecordSettings] = None
    ) -> BatchResults:
        """
        Run the provided batch by simulating each contained environment.

        :param batch: The batch to run.
        :param record_settings: Optional settings for recording the runnings. If None, no recording is made.
        :returns: List of simulation states in ascending order of time.
        """
        logging.info("Starting simulation batch with mujoco.")

        control_step = 1 / batch.control_frequency
        sample_step = 1 / batch.sampling_frequency

        if record_settings is not None:
            os.makedirs(record_settings.video_directory, exist_ok=False)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._num_simulators
        ) as executor:
            futures = [
                executor.submit(
                    self._run_environment,
                    env_index,
                    env_descr,
                    self._headless,
                    record_settings,
                    self._start_paused,
                    control_step,
                    sample_step,
                    batch.simulation_time,
                )
                for env_index, env_descr in enumerate(batch.environments)
            ]
            results = BatchResults([future.result() for future in futures])

        logging.info("Finished batch.")

        return results

    @staticmethod
    def _make_mjcf(env_descr: Environment,
                   amender: Union[None, Callable[[mjcf.RootElement], None]]) \
        -> str:

        env_mjcf = mjcf.RootElement(model="environment")

        env_mjcf.compiler.angle = "radian"

        env_mjcf.option.timestep = 0.002
        env_mjcf.option.integrator = "RK4"

        env_mjcf.option.gravity = [0, 0, -9.81]

        env_mjcf.worldbody.add(
            "geom",
            name="ground",
            type="plane",
            size=[10, 10, 1],
            rgba=[0.2, 0.2, 0.2, 1],
        )
        env_mjcf.worldbody.add(
            "light",
            pos=[0, 0, 100],
            ambient=[0.5, 0.5, 0.5],
            directional=True,
            castshadow=False,
        )
        env_mjcf.visual.headlight.active = 0

        for actor_index, posed_actor in enumerate(env_descr.actors):
            urdf = physbot_to_urdf(
                posed_actor.actor,
                f"robot_{actor_index}",
                Vector3(),
                Quaternion(),
            )

            model = mujoco.MjModel.from_xml_string(urdf)

            import tempfile
            import os
            # Create a temporary directory to store the temporary file
            temp_dir = tempfile.mkdtemp()
            # Specify the file path for the temporary XML file
            temp_file = os.path.join(temp_dir, "temp_mujoco.xml")
            # Save the XML to the temporary file
            mujoco.mj_saveLastXML(temp_file, model)
            # Load the XML from the temporary file
            robot = mjcf.from_file(temp_file)
            # Clean up the temporary directory and file
            os.remove(temp_file)
            os.rmdir(temp_dir)


            '''# mujoco can only save to a file, not directly to string,
            # so we create a temporary file.
            with tempfile.NamedTemporaryFile(
                mode="r+", delete=True, suffix="_mujoco.urdf"
            ) as botfile:
                mujoco.mj_saveLastXML(botfile.name, model)
                robot = mjcf.from_file(botfile)
            '''
            

            ctrl_range = 1.0    # What for?
            ctrl_range_str = f"-{ctrl_range} {ctrl_range}"
            force_range = 4.0   # limits force of actuator (prevent jumping)
            force_range_str = f"-{force_range} {force_range}"
            for joint in posed_actor.actor.joints:
                robot.find(namespace="joint", identifier=joint.name).armature = "0.2"
                robot.actuator.add(
                    "position",
                    # kp=5.0,
                    kp=10000,
                    ctrlrange=ctrl_range_str,
                    forcerange=force_range_str,
                    joint=robot.find(
                        namespace="joint",
                        identifier=joint.name,
                    ),
                )
                robot.actuator.add(
                    "velocity",
                    # kv=0.05,
                    kv=0.1,
                    ctrlrange=ctrl_range_str,
                    forcerange=force_range_str,
                    joint=robot.find(namespace="joint", identifier=joint.name),
                )

            # add a tracking camera (strangely does not work after attaching)
            dist = 1
            robot.worldbody.add("camera", name="tracker", mode="track",
                                dclass=robot.full_identifier,
                                pos=[-dist, 0, +dist],
                                euler=[0, -math.pi / 4., -math.pi/2])

            # Try to add fps camera to the front
            aabb = posed_actor.actor.calc_aabb()
            fps_cam_pos = [
                aabb.offset.x + aabb.size.x / 2,
                aabb.offset.y,
                aabb.offset.z
            ]
            robot.worldbody.add("camera", name="vision", mode="fixed", dclass=robot.full_identifier,
                                pos=fps_cam_pos, xyaxes="0 -1 0 0 0 1")
            robot.worldbody.add('site',
                                name=robot.full_identifier[:-2] + "_camera",
                                pos=fps_cam_pos, rgba=[0, 0, 1, 1],
                                type="ellipsoid", size=[0.0001, 0.025, 0.025])

            attachment_frame = env_mjcf.attach(robot)
            attachment_frame.add("freejoint")
            attachment_frame.pos = [*posed_actor.position]
            attachment_frame.quat = "1 0 0 0"  #[*posed_actor.orientation]  # TODO Rotation is flipped here
            # Mujoco default quaternion is [1,0,0,0] and pyrr is [0,0,0,1]

        if amender is not None:
            amender(env_mjcf)

        xml = env_mjcf.to_xml_string()
        # print(xml)
        if not isinstance(xml, str):
            raise RuntimeError("Error generating mjcf xml.")

        return xml

    @classmethod
    def _get_actor_states(
        cls, env_descr: Environment, data: mujoco.MjData, model: mujoco.MjModel
    ) -> List[ActorState]:
        return [
            cls._get_actor_state(i, data, model) for i in range(len(env_descr.actors))
        ]

    @staticmethod
    def _get_actor_state(
        robot_index: int, data: mujoco.MjData, model: mujoco.MjModel
    ) -> ActorState:
        bodyid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"robot_{robot_index}/",  # the slash is added by dm_control. ugly but deal with it
        )
        assert bodyid >= 0

        qindex = model.body_jntadr[bodyid]

        # explicitly copy because the Vector3 and Quaternion classes don't copy the underlying structure
        position = Vector3([n for n in data.qpos[qindex : qindex + 3]])
        orientation = Quaternion([n for n in data.qpos[qindex + 3 : qindex + 3 + 4]])

        return ActorState(position, orientation)

    @staticmethod
    def _set_dof_targets(data: mujoco.MjData, targets: List[float]) -> None:
        if len(targets) * 2 != len(data.ctrl):
            raise RuntimeError("Need to set a target for every dof")
        for i, target in enumerate(targets):
            data.ctrl[2 * i] = target
            data.ctrl[2 * i + 1] = 0
