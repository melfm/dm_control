# Copyright 2019 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A task where the goal is to move the hand close to a target prop or site."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import OrderedDict
import random

from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.manipulation.shared import arenas
from dm_control.manipulation.shared import cameras
from dm_control.manipulation.shared import constants
from dm_control.manipulation.shared import observations
from dm_control.manipulation.shared import registry
from dm_control.manipulation.shared import robots
from dm_control.manipulation.shared import tags
from dm_control.manipulation.shared import workspaces
from dm_control.utils import rewards
import numpy as np

_ReachWorkspace = collections.namedtuple(
    '_ReachWorkspace', ['target_bbox', 'tcp_bbox', 'arm_offset'])

# Ensures that the props are not touching the table before settling.
_PROP_Z_OFFSET = 0.001

_DUPLO_WORKSPACE = _ReachWorkspace(
    target_bbox=workspaces.BoundingBox(lower=(-0.1, -0.1, _PROP_Z_OFFSET),
                                       upper=(0.1, 0.1, _PROP_Z_OFFSET)),
    tcp_bbox=workspaces.BoundingBox(lower=(-0.1, -0.1, 0.2),
                                    upper=(0.1, 0.1, 0.4)),
    arm_offset=robots.ARM_OFFSET)

# Fixing the target for single-goal reach
# X:Left/Right+   Y:Up/Down   Z:Forward/Backward
_SITE_WORKSPACE = _ReachWorkspace(
    target_bbox=workspaces.BoundingBox(lower=(0.23, 0.12, 0.02),
                                       upper=(0.25, 0.15, 0.05)),
    tcp_bbox=workspaces.BoundingBox(lower=(0.1, 0.1, 0.02),
                                    upper=(0.1, 0.1, 0.02)),
    arm_offset=robots.ARM_OFFSET)

# _SITE_WORKSPACE = _ReachWorkspace(
#     target_bbox=workspaces.BoundingBox(lower=(-0.2, -0.2, 0.02),
#                                        upper=(0.2, 0.2, 0.4)),
#     tcp_bbox=workspaces.BoundingBox(lower=(-0.2, -0.2, 0.02),
#                                     upper=(0.2, 0.2, 0.4)),
#     arm_offset=robots.ARM_OFFSET)

_TARGET_RADIUS = 0.05


class EnvRandomizer():
    """EnvRandomizer Class used for visually randomizing the environment.
    """
    def __init__(self):

        self.table_tags = OrderedDict([('table_dark_wood', 0),
                                       ('table_marble', 1), ('table_blue', 2),
                                       ('table_tennis', 3), ('table_wood', 4),
                                       ('table_light_wood_v3', 5),
                                       ('table_light_wood_v2', 6),
                                       ('default', 7)])

        self.table_eval_tags = OrderedDict([('robot', 0), ('robot_bw', 1),
                                            ('real_desk', 2), ('default', 3)])
        # Not used atm
        self.goal_tags = OrderedDict([('goal_red', 0), ('goal_yellow', 1),
                                      ('goal_blue', 2), ('goal_pink', 3),
                                      ('default', 4)])

        self.sky_tags = OrderedDict([('default', 0), ('red_star', 1),
                                     ('orange_star', 2), ('yellow_star', 3),
                                     ('pink_star', 4), ('amber_star', 5),
                                     ('white_star', 6), ('black_star', 7)])

        self.eval_env_tags = OrderedDict([('table_granite_goal_purple', 0)])


class Reach(composer.Task):
    """Bring the hand close to a target prop or site."""
    def __init__(self, arena, arm, hand, prop, obs_settings, workspace,
                 control_timestep, table_col_tag, sky_col_tag):
        """Initializes a new `Reach` task.

    Args:
      arena: `composer.Entity` instance.
      arm: `robot_base.RobotArm` instance.
      hand: `robot_base.RobotHand` instance.
      prop: `composer.Entity` instance specifying the prop to reach to, or None
        in which case the target is a fixed site whose position is specified by
        the workspace.
      obs_settings: `observations.ObservationSettings` instance.
      workspace: `_ReachWorkspace` specifying the placement of the prop and TCP.
      control_timestep: Float specifying the control timestep in seconds.
    """
        self._arena = arena
        self._arm = arm
        self._hand = hand
        self._arm.attach(self._hand)
        self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
        self.control_timestep = control_timestep
        self._tcp_initializer = initializers.ToolCenterPointInitializer(
            self._hand,
            self._arm,
            position=distributions.Uniform(*workspace.tcp_bbox),
            quaternion=workspaces.DOWN_QUATERNION)

        # Add custom camera observable.
        self._task_observables = cameras.add_camera_observables(
            arena, obs_settings, cameras.FRONT_FAR)

        target_pos_distribution = distributions.Uniform(*workspace.target_bbox)
        self._prop = prop
        if prop:
            # The prop itself is used to visualize the target location.
            self._make_target_site(parent_entity=prop, visible=False)
            self._target = self._arena.add_free_entity(prop)
            self._prop_placer = initializers.PropPlacer(
                props=[prop],
                position=target_pos_distribution,
                quaternion=workspaces.uniform_z_rotation,
                settle_physics=True)
        else:
            self._target = self._make_target_site(parent_entity=arena,
                                                  visible=True)
            self._target_placer = target_pos_distribution

            obs = observable.MJCFFeature('pos', self._target)
            obs.configure(**obs_settings.prop_pose._asdict())
            self._task_observables['target_position'] = obs

        # Randomize the table surface
        if table_col_tag == 0:
            self.root_entity.mjcf_model.worldbody.add('geom',
                                                      type='plane',
                                                      pos="0 0 0.01",
                                                      size="0.6 0.6 0.5",
                                                      rgba=".6 .6 .5 1",
                                                      contype="1",
                                                      conaffinity="1",
                                                      friction="2 0.1 0.002",
                                                      material="j2s7/darkwood")
        elif table_col_tag == 1:
            self.root_entity.mjcf_model.worldbody.add('geom',
                                                      type='plane',
                                                      pos="0 0 0.01",
                                                      size="0.6 0.6 0.5",
                                                      rgba=".6 .6 .5 1",
                                                      contype="1",
                                                      conaffinity="1",
                                                      friction="2 0.1 0.002",
                                                      material="j2s7/marble")
        elif table_col_tag == 2:
            self.root_entity.mjcf_model.worldbody.add(
                'geom',
                type='plane',
                pos="0 0 0.01",
                size="0.6 0.6 0.5",
                rgba=".6 .6 .5 1",
                contype="1",
                conaffinity="1",
                friction="2 0.1 0.002",
                material="j2s7/navy_blue")
        elif table_col_tag == 3:
            self.root_entity.mjcf_model.worldbody.add('geom',
                                                      type='plane',
                                                      pos="0 0 0.01",
                                                      size="0.6 0.6 0.5",
                                                      rgba=".6 .6 .5 1",
                                                      contype="1",
                                                      conaffinity="1",
                                                      friction="2 0.1 0.002",
                                                      material="j2s7/tennis")
        elif table_col_tag == 4:
            self.root_entity.mjcf_model.worldbody.add('geom',
                                                      type='plane',
                                                      pos="0 0 0.01",
                                                      size="0.6 0.6 0.5",
                                                      rgba=".6 .6 .5 1",
                                                      contype="1",
                                                      conaffinity="1",
                                                      friction="2 0.1 0.002",
                                                      material="j2s7/wood")
        elif table_col_tag == 5:
            self.root_entity.mjcf_model.worldbody.add(
                'geom',
                type='plane',
                pos="0 0 0.01",
                size="0.6 0.6 0.5",
                rgba=".6 .6 .5 1",
                contype="1",
                conaffinity="1",
                friction="2 0.1 0.002",
                material="j2s7/light_wood_v3")

        elif table_col_tag == 6:
            self.root_entity.mjcf_model.worldbody.add(
                'geom',
                type='plane',
                pos="0 0 0.01",
                size="0.6 0.6 0.5",
                rgba=".6 .6 .5 1",
                contype="1",
                conaffinity="1",
                friction="2 0.1 0.002",
                material="j2s7/light_wood_v2")

        # Sky change of colour
        if sky_col_tag == 1:
            # Red stary sky
            self.root_entity.mjcf_model.asset.texture[0].mark = 'random'
            self.root_entity.mjcf_model.asset.texture[0].rgb1 = np.array(
                [.8, .1, .4])
            self.root_entity.mjcf_model.asset.texture[0].width = 800
            self.root_entity.mjcf_model.asset.texture[0].height = 800

        elif sky_col_tag == 2:
            # Orange stary sky
            self.root_entity.mjcf_model.asset.texture[0].mark = 'random'
            self.root_entity.mjcf_model.asset.texture[0].rgb1 = np.array(
                [.8, .5, .1])
            self.root_entity.mjcf_model.asset.texture[0].width = 800
            self.root_entity.mjcf_model.asset.texture[0].height = 800

        elif sky_col_tag == 3:
            # Yellow stary sky
            self.root_entity.mjcf_model.asset.texture[0].mark = 'random'
            self.root_entity.mjcf_model.asset.texture[0].rgb1 = np.array(
                [1, 1, .4])
            self.root_entity.mjcf_model.asset.texture[0].width = 800
            self.root_entity.mjcf_model.asset.texture[0].height = 800

        elif sky_col_tag == 4:
            # Pink stary sky
            self.root_entity.mjcf_model.asset.texture[0].mark = 'random'
            self.root_entity.mjcf_model.asset.texture[0].rgb1 = np.array(
                [1, .5, 1])
            self.root_entity.mjcf_model.asset.texture[0].width = 800
            self.root_entity.mjcf_model.asset.texture[0].height = 800

        elif sky_col_tag == 5:
            # Amber stary sky
            self.root_entity.mjcf_model.asset.texture[0].mark = 'random'
            self.root_entity.mjcf_model.asset.texture[0].rgb1 = np.array(
                [1, .6, .4])
            self.root_entity.mjcf_model.asset.texture[0].width = 800
            self.root_entity.mjcf_model.asset.texture[0].height = 800

        elif sky_col_tag == 6:
            # White stary sky
            self.root_entity.mjcf_model.asset.texture[0].rgb1 = np.array(
                [1., 1., 1.])
            self.root_entity.mjcf_model.asset.texture[0].rgb2 = np.array(
                [1., 1., 1.])
            self.root_entity.mjcf_model.asset.texture[0].width = 800
            self.root_entity.mjcf_model.asset.texture[0].height = 800

        elif sky_col_tag == 7:
            # Black stary sky
            self.root_entity.mjcf_model.asset.texture[0].rgb1 = np.array(
                [0, 0, 0])
            self.root_entity.mjcf_model.asset.texture[0].width = 800
            self.root_entity.mjcf_model.asset.texture[0].height = 800

        # TODO: Remove the checkerboard groundplane
        # For now this somehow sets it to a white plane!
        self.root_entity.mjcf_model.asset.texture[1].width = 1
        self.root_entity.mjcf_model.asset.texture[1].height = 1
        # Add sites for visualizing the prop and target bounding boxes.
        workspaces.add_bbox_site(body=self.root_entity.mjcf_model.worldbody,
                                 lower=workspace.tcp_bbox.lower,
                                 upper=workspace.tcp_bbox.upper,
                                 rgba=constants.GREEN,
                                 name='tcp_spawn_area')
        workspaces.add_bbox_site(body=self.root_entity.mjcf_model.worldbody,
                                 lower=workspace.target_bbox.lower,
                                 upper=workspace.target_bbox.upper,
                                 rgba=constants.BLUE,
                                 name='target_spawn_area')

    def _make_target_site(self, parent_entity, visible):
        return workspaces.add_target_site(
            body=parent_entity.mjcf_model.worldbody,
            radius=_TARGET_RADIUS,
            visible=visible,
            rgba=constants.RED,
            name='target_site')

    @property
    def root_entity(self):
        return self._arena

    @property
    def arm(self):
        return self._arm

    @property
    def hand(self):
        return self._hand

    @property
    def task_observables(self):
        return self._task_observables

    # def get_reward_sparse(self, physics):
    #   hand_pos = physics.bind(self._hand.tool_center_point).xpos
    #   target_pos = physics.bind(self._target).xpos
    #   distance = np.linalg.norm(hand_pos - target_pos)
    #   return rewards.tolerance(
    #       distance, bounds=(0, _TARGET_RADIUS), margin=_TARGET_RADIUS)

    def get_reward(self, physics):
        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        hand_pos = physics.bind(self._hand.tool_center_point).xpos
        target_pos = physics.bind(self._target).xpos
        distance = np.linalg.norm(hand_pos - target_pos)
        reachRew = c1 * (self.maxReachDist - distance) + c1 * (
            np.exp(-(distance**2) / c2) + np.exp(-(distance**2) / c3))
        reachRew = max(reachRew, 0)
        return reachRew

    def initialize_episode(self, physics, random_state):
        self._hand.set_grasp(physics, close_factors=random_state.uniform())
        self._tcp_initializer(physics, random_state)
        init_hand_pos = physics.bind(self._hand.tool_center_point).xpos
        self.target_pos = physics.bind(self._target).xpos
        self.maxReachDist = np.linalg.norm(init_hand_pos - self.target_pos)
        if self._prop:
            self._prop_placer(physics, random_state)
        else:
            physics.bind(self._target).pos = (self._target_placer(
                random_state=random_state))


class ReachEval(composer.Task):
    """Bring the hand close to a target prop or site."""
    def __init__(self, arena, arm, hand, prop, obs_settings, workspace,
                 control_timestep, table_col_tag):
        """Initializes a new `Reach` task.

    Args:
      arena: `composer.Entity` instance.
      arm: `robot_base.RobotArm` instance.
      hand: `robot_base.RobotHand` instance.
      prop: `composer.Entity` instance specifying the prop to reach to, or None
        in which case the target is a fixed site whose position is specified by
        the workspace.
      obs_settings: `observations.ObservationSettings` instance.
      workspace: `_ReachWorkspace` specifying the placement of the prop and TCP.
      control_timestep: Float specifying the control timestep in seconds.
    """
        self._arena = arena
        self._arm = arm
        self._hand = hand
        self._arm.attach(self._hand)
        self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
        self.control_timestep = control_timestep
        self._tcp_initializer = initializers.ToolCenterPointInitializer(
            self._hand,
            self._arm,
            position=distributions.Uniform(*workspace.tcp_bbox),
            quaternion=workspaces.DOWN_QUATERNION)

        # Add custom camera observable.
        self._task_observables = cameras.add_camera_observables(
            arena, obs_settings, cameras.FRONT_FAR)

        target_pos_distribution = distributions.Uniform(*workspace.target_bbox)
        self._prop = prop
        if prop:
            # The prop itself is used to visualize the target location.
            self._make_target_site(parent_entity=prop, visible=False)
            self._target = self._arena.add_free_entity(prop)
            self._prop_placer = initializers.PropPlacer(
                props=[prop],
                position=target_pos_distribution,
                quaternion=workspaces.uniform_z_rotation,
                settle_physics=True)
        else:
            self._target = self._make_target_site(parent_entity=arena,
                                                  visible=True)
            self._target_placer = target_pos_distribution

            obs = observable.MJCFFeature('pos', self._target)
            obs.configure(**obs_settings.prop_pose._asdict())
            self._task_observables['target_position'] = obs

        # Randomize the table surface
        if table_col_tag == 0:
            self.root_entity.mjcf_model.worldbody.add('geom',
                                                      type='plane',
                                                      pos="0 0 0.01",
                                                      size="0.6 0.6 0.5",
                                                      rgba=".6 .6 .5 1",
                                                      contype="1",
                                                      conaffinity="1",
                                                      friction="2 0.1 0.002",
                                                      material="j2s7/robot")
        elif table_col_tag == 1:
            self.root_entity.mjcf_model.worldbody.add('geom',
                                                      type='plane',
                                                      pos="0 0 0.01",
                                                      size="0.6 0.6 0.5",
                                                      rgba=".6 .6 .5 1",
                                                      contype="1",
                                                      conaffinity="1",
                                                      friction="2 0.1 0.002",
                                                      material="j2s7/robot_bw")

        elif table_col_tag == 2:
            self.root_entity.mjcf_model.worldbody.add('geom',
                                                      type='plane',
                                                      pos="0 0 0.01",
                                                      size="0.6 0.6 0.5",
                                                      rgba=".6 .6 .5 1",
                                                      contype="1",
                                                      conaffinity="1",
                                                      friction="2 0.1 0.002",
                                                      material="j2s7/real_desk")

        # Blue stary sky
        self.root_entity.mjcf_model.asset.texture[0].mark = 'random'
        self.root_entity.mjcf_model.asset.texture[0].rgb1 = np.array(
            [.4, .6, .8])
        self.root_entity.mjcf_model.asset.texture[0].width = 800
        self.root_entity.mjcf_model.asset.texture[0].height = 800

        # How to remove the checkerboard groundplane ?
        # For now this somehow sets it to a white plane!
        self.root_entity.mjcf_model.asset.texture[1].width = 1
        self.root_entity.mjcf_model.asset.texture[1].height = 1
        # Add sites for visualizing the prop and target bounding boxes.
        workspaces.add_bbox_site(body=self.root_entity.mjcf_model.worldbody,
                                 lower=workspace.tcp_bbox.lower,
                                 upper=workspace.tcp_bbox.upper,
                                 rgba=constants.GREEN,
                                 name='tcp_spawn_area')
        workspaces.add_bbox_site(body=self.root_entity.mjcf_model.worldbody,
                                 lower=workspace.target_bbox.lower,
                                 upper=workspace.target_bbox.upper,
                                 rgba=constants.BLUE,
                                 name='target_spawn_area')

    def _make_target_site(self, parent_entity, visible):
        return workspaces.add_target_site(
            body=parent_entity.mjcf_model.worldbody,
            radius=_TARGET_RADIUS,
            visible=visible,
            rgba=constants.RED,
            name='target_site')

    @property
    def root_entity(self):
        return self._arena

    @property
    def arm(self):
        return self._arm

    @property
    def hand(self):
        return self._hand

    @property
    def task_observables(self):
        return self._task_observables

    # def get_reward_sparse(self, physics):
    #   hand_pos = physics.bind(self._hand.tool_center_point).xpos
    #   target_pos = physics.bind(self._target).xpos
    #   distance = np.linalg.norm(hand_pos - target_pos)
    #   return rewards.tolerance(
    #       distance, bounds=(0, _TARGET_RADIUS), margin=_TARGET_RADIUS)

    def get_reward(self, physics):
        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        hand_pos = physics.bind(self._hand.tool_center_point).xpos
        target_pos = physics.bind(self._target).xpos
        distance = np.linalg.norm(hand_pos - target_pos)
        reachRew = c1 * (self.maxReachDist - distance) + c1 * (
            np.exp(-(distance**2) / c2) + np.exp(-(distance**2) / c3))
        reachRew = max(reachRew, 0)
        return reachRew

    def initialize_episode(self, physics, random_state):
        self._hand.set_grasp(physics, close_factors=random_state.uniform())
        self._tcp_initializer(physics, random_state)
        init_hand_pos = physics.bind(self._hand.tool_center_point).xpos
        self.target_pos = physics.bind(self._target).xpos
        self.maxReachDist = np.linalg.norm(init_hand_pos - self.target_pos)
        if self._prop:
            self._prop_placer(physics, random_state)
        else:
            physics.bind(self._target).pos = (self._target_placer(
                random_state=random_state))


def _reach(obs_settings, use_site, evalenv=False):
    """Configure and instantiate a `Reach` task.

  Args:
    obs_settings: An `observations.ObservationSettings` instance.
    use_site: Boolean, if True then the target will be a fixed site, otherwise
      it will be a moveable Duplo brick.

  Returns:
    An instance of `reach.Reach`.
  """
    arena = arenas.Standard()
    arm = robots.make_arm(obs_settings=obs_settings)
    hand = robots.make_hand(obs_settings=obs_settings)
    if use_site:
        workspace = _SITE_WORKSPACE
        prop = None
    else:
        workspace = _DUPLO_WORKSPACE
        prop = props.Duplo(observable_options=observations.make_options(
            obs_settings, observations.FREEPROP_OBSERVABLES))

    env_randomizer = EnvRandomizer()
    table_col_tag = random.choice(list(env_randomizer.table_tags.values()))
    table_env_col_tag = random.choice(
        list(env_randomizer.table_eval_tags.values()))
    sky_col_tag = random.choice(list(env_randomizer.sky_tags.values()))

    if evalenv:
        task = ReachEval(arena=arena,
                         arm=arm,
                         hand=hand,
                         prop=prop,
                         obs_settings=obs_settings,
                         workspace=workspace,
                         control_timestep=constants.CONTROL_TIMESTEP,
                         table_col_tag=table_env_col_tag)
    else:
        task = Reach(arena=arena,
                     arm=arm,
                     hand=hand,
                     prop=prop,
                     obs_settings=obs_settings,
                     workspace=workspace,
                     control_timestep=constants.CONTROL_TIMESTEP,
                     table_col_tag=table_col_tag,
                     sky_col_tag=sky_col_tag)
    mod_id = str(table_col_tag) + str(sky_col_tag) + str(
        random.randint(10, 1000))
    return task, mod_id


@registry.add(tags.FEATURES, tags.EASY)
def reach_duplo_features():
    return _reach(obs_settings=observations.PERFECT_FEATURES, use_site=False)


@registry.add(tags.VISION, tags.EASY)
def reach_duplo_vision():
    return _reach(obs_settings=observations.VISION, use_site=False)


@registry.add(tags.FEATURES, tags.EASY)
def reach_site_features():
    task, tag = _reach(obs_settings=observations.PERFECT_FEATURES,
                       use_site=True,
                       evalenv=False)
    return task, tag


@registry.add(tags.FEATURES, tags.EASY)
def reach_site_features_eval():
    task, tag = _reach(obs_settings=observations.PERFECT_FEATURES,
                       use_site=True,
                       evalenv=True)
    return task, tag


@registry.add(tags.VISION, tags.EASY)
def reach_site_vision():
    return _reach(obs_settings=observations.VISION, use_site=True)