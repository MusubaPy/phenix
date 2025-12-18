# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
add_custom_target(copy_model_resources ALL)

# Helper macro to copy and optionally patch a file from SOURCE to DEST.
macro(_copy_and_patch src_file dest_dir dest_file src_patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${CMAKE_COMMAND} -E make_directory ${dest_dir}
                COMMAND ${CMAKE_COMMAND} -E copy ${src_file} ${dest_file}
        )
        if(EXISTS ${src_patch})
                add_custom_command(TARGET copy_model_resources
                        COMMAND ${PATCH_EXECUTABLE} -i ${src_patch} -o ${dest_file}.modified ${dest_file}
                        VERBATIM
                )
        endif()
endmacro()

find_program(PATCH_EXECUTABLE patch)
if(NOT PATCH_EXECUTABLE)
        message(STATUS "'patch' not found in PATH; patch steps will be skipped where required")
endif()

# dm_control models
_copy_and_patch(${dm_control_SOURCE_DIR}/dm_control/suite/acrobot.xml
                                                         ${CMAKE_CURRENT_BINARY_DIR}/acrobot
                                                         ${CMAKE_CURRENT_BINARY_DIR}/acrobot/acrobot.xml
                                                         ${CMAKE_CURRENT_SOURCE_DIR}/acrobot/acrobot.xml.patch)

_copy_and_patch(${dm_control_SOURCE_DIR}/dm_control/suite/cartpole.xml
                                                         ${CMAKE_CURRENT_BINARY_DIR}/cartpole
                                                         ${CMAKE_CURRENT_BINARY_DIR}/cartpole/cartpole.xml
                                                         ${CMAKE_CURRENT_SOURCE_DIR}/cartpole/cartpole.xml.patch)

_copy_and_patch(${dm_control_SOURCE_DIR}/dm_control/suite/humanoid.xml
                                                         ${CMAKE_CURRENT_BINARY_DIR}/humanoid
                                                         ${CMAKE_CURRENT_BINARY_DIR}/humanoid/humanoid.xml
                                                         ${CMAKE_CURRENT_SOURCE_DIR}/humanoid/humanoid.xml.patch)

_copy_and_patch(${dm_control_SOURCE_DIR}/dm_control/suite/point_mass.xml
                                                         ${CMAKE_CURRENT_BINARY_DIR}/particle
                                                         ${CMAKE_CURRENT_BINARY_DIR}/particle/particle.xml
                                                         ${CMAKE_CURRENT_SOURCE_DIR}/particle/particle.xml.patch)

_copy_and_patch(${dm_control_SOURCE_DIR}/dm_control/suite/swimmer.xml
                                                         ${CMAKE_CURRENT_BINARY_DIR}/swimmer
                                                         ${CMAKE_CURRENT_BINARY_DIR}/swimmer/swimmer.xml
                                                         ${CMAKE_CURRENT_SOURCE_DIR}/swimmer/swimmer.xml.patch)

_copy_and_patch(${dm_control_SOURCE_DIR}/dm_control/suite/walker.xml
                                                         ${CMAKE_CURRENT_BINARY_DIR}/walker
                                                         ${CMAKE_CURRENT_BINARY_DIR}/walker/walker.xml
                                                         ${CMAKE_CURRENT_SOURCE_DIR}/walker/walker.xml.patch)

# Menagerie models
add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shadow_reorient
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/shadow_hand/right_hand.xml ${CMAKE_CURRENT_BINARY_DIR}/shadow_reorient/right_hand.xml
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${menagerie_SOURCE_DIR}/shadow_hand/assets ${CMAKE_CURRENT_BINARY_DIR}/shadow_reorient/assets
)

add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/allegro
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/wonik_allegro/right_hand.xml ${CMAKE_CURRENT_BINARY_DIR}/allegro/right_hand.xml
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${menagerie_SOURCE_DIR}/wonik_allegro/assets ${CMAKE_CURRENT_BINARY_DIR}/allegro/assets
)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/allegro/right_hand.xml.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_SOURCE_DIR}/allegro/right_hand.xml.patch -o ${CMAKE_CURRENT_BINARY_DIR}/allegro/right_hand.xml.modified ${CMAKE_CURRENT_BINARY_DIR}/allegro/right_hand.xml
        )
endif()

add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/panda
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/franka_emika_panda/panda.xml ${CMAKE_CURRENT_BINARY_DIR}/panda/panda.xml
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${menagerie_SOURCE_DIR}/franka_emika_panda/assets ${CMAKE_CURRENT_BINARY_DIR}/panda/assets
)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/panda/panda.xml.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_SOURCE_DIR}/panda/panda.xml.patch -o ${CMAKE_CURRENT_BINARY_DIR}/panda/panda.xml.modified ${CMAKE_CURRENT_BINARY_DIR}/panda/panda.xml
        )
endif()

add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/quadruped_mod
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/unitree_a1/a1.xml ${CMAKE_CURRENT_BINARY_DIR}/quadruped_mod/a1.xml
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${menagerie_SOURCE_DIR}/unitree_a1/assets ${CMAKE_CURRENT_BINARY_DIR}/quadruped_mod/assets
)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/quadruped_mod/a1.xml.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_SOURCE_DIR}/quadruped_mod/a1.xml.patch -o ${CMAKE_CURRENT_BINARY_DIR}/quadruped_mod/a1.xml.modified ${CMAKE_CURRENT_BINARY_DIR}/quadruped_mod/a1.xml
        )
endif()

add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/quadruped_vanila
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/unitree_a1/a1.xml ${CMAKE_CURRENT_BINARY_DIR}/quadruped_vanila/a1.xml
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${menagerie_SOURCE_DIR}/unitree_a1/assets ${CMAKE_CURRENT_BINARY_DIR}/quadruped_vanila/assets
)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/quadruped_vanila/a1.xml.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_SOURCE_DIR}/quadruped_vanila/a1.xml.patch -o ${CMAKE_CURRENT_BINARY_DIR}/quadruped_vanila/a1.xml.modified ${CMAKE_CURRENT_BINARY_DIR}/quadruped_vanila/a1.xml
        )
endif()

add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${menagerie_SOURCE_DIR}/franka_emika_panda ${CMAKE_CURRENT_BINARY_DIR}/manipulation
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${menagerie_SOURCE_DIR}/robotiq_2f85 ${CMAKE_CURRENT_BINARY_DIR}/manipulation
)

add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/quadrotor
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/skydio_x2/x2.xml ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/quadrotor.xml
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${menagerie_SOURCE_DIR}/skydio_x2/assets ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/assets
)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/quadrotor/quadrotor.xml.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_SOURCE_DIR}/quadrotor/quadrotor.xml.patch -o ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/quadrotor.xml.modified ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/quadrotor.xml
        )
endif()

# Cube reorientation
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/common_assets/cube.xml.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_SOURCE_DIR}/common_assets/cube.xml.patch -o ${CMAKE_CURRENT_BINARY_DIR}/common_assets/cube_modified_shadow_reorient.xml ${CMAKE_CURRENT_BINARY_DIR}/common_assets/reorientation_cube.xml
        )
endif()

# Cube solve task
add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/rubik
        COMMAND ${CMAKE_COMMAND} -E copy ${mujoco_SOURCE_DIR}/model/cube/cube_3x3x3.xml ${CMAKE_CURRENT_BINARY_DIR}/rubik/cube_3x3x3.xml
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${mujoco_SOURCE_DIR}/model/cube/assets ${CMAKE_CURRENT_BINARY_DIR}/rubik/assets
)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/rubik/cube_3x3x3.xml.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_SOURCE_DIR}/rubik/cube_3x3x3.xml.patch -o ${CMAKE_CURRENT_BINARY_DIR}/rubik/cube_3x3x3_modified.xml ${CMAKE_CURRENT_BINARY_DIR}/rubik/cube_3x3x3.xml
        )
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/rubik/transition_model.xml.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_SOURCE_DIR}/rubik/transition_model.xml.patch -o ${CMAKE_CURRENT_BINARY_DIR}/rubik/transition_model.xml ${CMAKE_CURRENT_BINARY_DIR}/rubik/cube_3x3x3.xml
        )
endif()

add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/shadow_hand/right_hand.xml ${CMAKE_CURRENT_BINARY_DIR}/rubik/right_hand.xml
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${menagerie_SOURCE_DIR}/shadow_hand/assets ${CMAKE_CURRENT_BINARY_DIR}/rubik/assets
)

# ALOHA
add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${menagerie_SOURCE_DIR}/aloha/assets ${CMAKE_CURRENT_BINARY_DIR}/bimanual/assets
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/aloha/aloha.xml ${CMAKE_CURRENT_BINARY_DIR}/bimanual/aloha.xml
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/aloha/filtered_cartesian_actuators.xml ${CMAKE_CURRENT_BINARY_DIR}/bimanual/filtered_cartesian_actuators.xml
)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bimanual/aloha.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_SOURCE_DIR}/bimanual/aloha.patch -o ${CMAKE_CURRENT_BINARY_DIR}/bimanual/aloha_cartesian.xml ${CMAKE_CURRENT_BINARY_DIR}/bimanual/aloha.xml
        )
endif()

add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/aloha/mjx_aloha.patch ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_aloha.patch
)
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_aloha.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_aloha.patch -o ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_aloha.xml ${CMAKE_CURRENT_BINARY_DIR}/bimanual/aloha.xml
        )
endif()

add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/aloha/scene.xml ${CMAKE_CURRENT_BINARY_DIR}/bimanual/scene.xml
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/aloha/mjx_scene.patch ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_scene.patch
)
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_scene.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_scene.patch -o ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_scene.xml ${CMAKE_CURRENT_BINARY_DIR}/bimanual/scene.xml
        )
endif()

add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/aloha/mjx_filtered_cartesian_actuators.patch ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_filtered_cartesian_actuators.patch
)
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_filtered_cartesian_actuators.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_filtered_cartesian_actuators.patch -o ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_filtered_cartesian_actuators.xml ${CMAKE_CURRENT_BINARY_DIR}/bimanual/filtered_cartesian_actuators.xml
        )
endif()

add_custom_command(TARGET copy_model_resources
        COMMAND ${CMAKE_COMMAND} -E copy ${menagerie_SOURCE_DIR}/robotis_op3/op3.xml ${CMAKE_CURRENT_BINARY_DIR}/op3/op3.xml
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${menagerie_SOURCE_DIR}/robotis_op3/assets ${CMAKE_CURRENT_BINARY_DIR}/op3/assets
)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op3/op3.xml.patch)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${PATCH_EXECUTABLE} -i ${CMAKE_CURRENT_SOURCE_DIR}/op3/op3.xml.patch -o ${CMAKE_CURRENT_BINARY_DIR}/op3/op3_modified.xml ${CMAKE_CURRENT_BINARY_DIR}/op3/op3.xml
        )
endif()

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/manipulation/merge_panda_robotiq.py)
        add_custom_command(TARGET copy_model_resources
                COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/manipulation/merge_panda_robotiq.py ${CMAKE_CURRENT_BINARY_DIR}/manipulation/panda_robotiq.xml
                COMMENT "Merging panda and robotiq manipulation models"
        )
else()
        add_custom_command(TARGET copy_model_resources
                COMMENT "Skipping panda+robotiq merge: merge_panda_robotiq.py not found"
        )
endif()

add_custom_target(copy_resources ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Copying tasks into binary directory")

add_dependencies(copy_model_resources copy_resources)

add_dependencies(libmjpc copy_model_resources)
