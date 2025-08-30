import yourdfpy
import viser
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf
from viser import transforms as tf
import time
import numpy as np

import frantik

def se3_from_pos_xyzw(position, quat_xyzw):
    se3 = np.eye(4)
    se3[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    se3[:3, 3] = np.asarray(position)
    return se3

def compute_ik(position, quat_wxyz):
    w, x, y, z = quat_wxyz
    quat_xyzw = np.array([x, y, z, w])
    se3 = se3_from_pos_xyzw(position, quat_xyzw)
    return frantik.ik(se3.flatten().tolist(), 0, [0, 0, 0, 0, 0, 0, 0])

def create_viser_stick_gripper(server, name = "/stick_gripper", scale = 1.0, line_width = 5.0):
    finger_offset_x = 0.041 * scale
    finger_height = (0.11217 - 0.0659999996) * scale
    finger_base_z = 0.0659999996 * scale
    stick_height = finger_base_z
    bar_length = 0.082 * scale

    line_segments = []
    colors = []

    # Gripper from origin to base of claw
    line_segments.append([[0, 0, 0], [0, 0, stick_height]])
    colors.append([[0, 0, 255], [0, 0, 255]])

    line_segments.append(
        [[finger_offset_x, 0, finger_base_z], [finger_offset_x, 0, finger_base_z + finger_height]]
        )
    colors.append([[0, 0, 255], [0, 0, 255]])

    line_segments.append(
        [[-finger_offset_x, 0, finger_base_z], [-finger_offset_x, 0, finger_base_z + finger_height]]
        )
    colors.append([[0, 0, 255], [0, 0, 255]])

    line_segments.append([[-bar_length / 2, 0, finger_base_z], [bar_length / 2, 0, finger_base_z]])
    colors.append([[0, 0, 255], [0, 0, 255]])

    points = np.array(line_segments, dtype = np.float32)
    colors = np.array(colors, dtype = np.uint8)

    gripper_handle = server.scene.add_line_segments(
        name = name, points = points, colors = colors, line_width = line_width
        )

    return gripper_handle


def control_ee(server, panda):
    insert_grasp = server.gui.add_button("Insert Grasp")

    @insert_grasp.on_click
    def ik_and_plan(event):
        stick_gripper = create_viser_stick_gripper(server)
        insert_grasp.remove()
        cancel_plan = server.gui.add_button("Cancel Grasp Planning")
        solve_ik = server.gui.add_button("Solve IK")
        plan = server.gui.add_button("Plan motion")
        plan.visible = False

        xyz_input = server.gui.add_vector3(
            "Position (XYZ) meters",
            initial_value = (0.47, 0.0, 0.45),
            min = (-10.0, -10.0, -10.0),
            max = (10.0, 10.0, 10.0),
            step = 0.01
            )
        rpy_input = server.gui.add_vector3(
            "Rotation (RPY) radians",
            initial_value = (3.14, 0.0, 1.59),
            min = (-3.14159, -3.14159, -3.14159),
            max = (3.14159, 3.14159, 3.14159),
            step = 0.01
            )
        position = xyz_input.value
        rpy = rpy_input.value
        wxyz = tf.SO3.from_rpy_radians(rpy[0], rpy[1], rpy[2]).wxyz
        stick_gripper.position = position
        stick_gripper.wxyz = wxyz

        panda_shadow = None

        @xyz_input.on_update
        def update_gripper_pose(event):
            position = xyz_input.value
            rpy = rpy_input.value
            wxyz = tf.SO3.from_rpy_radians(rpy[0], rpy[1], rpy[2]).wxyz
            stick_gripper.position = position
            stick_gripper.wxyz = wxyz

        @rpy_input.on_update
        def update_gripper_rotation(event):
            update_gripper_pose(event)

        @solve_ik.on_click
        def solve_ik(event):
            cancel_plan.visible = False
            solve_ik.visible = False
            xyz_input.visible = False
            rpy_input.visible = False
            progress_message = server.gui.add_markdown("**Solving IK...**")
            nonlocal panda_shadow
            position = stick_gripper.position
            orientation = stick_gripper.wxyz

            config = compute_ik(position, orientation)
            print(config)
            urdf = yourdfpy.URDF.load("assets/panda/panda_spherized.urdf")
            panda_shadow = ViserUrdf(
                server,
                urdf,
                load_meshes = True,
                load_collision_meshes = False,
                root_node_name = "/panda_shadow",
                mesh_color_override = (0, 255, 0, 0.8)
                )
            panda_shadow.update_cfg(np.array(config))
            progress_message.remove()
            cancel_plan.visible = True
            solve_ik.visible = True
            plan.visible = True
            xyz_input.visible = True
            rpy_input.visible = True

        @cancel_plan.on_click
        def cancel_grasp(event):
            cancel_plan.remove()
            plan.remove()
            solve_ik.remove()
            xyz_input.remove()
            rpy_input.remove()
            stick_gripper.remove()
            if panda_shadow is not None:
                panda_shadow.remove()
            # Recreate the original "Insert Grasp" button
            nonlocal insert_grasp
            insert_grasp = server.gui.add_button("Insert Grasp")
            insert_grasp.on_click(ik_and_plan)


def main():
    server = viser.ViserServer()

    floor_grid = server.scene.add_grid(
        name = "/floor_grid",
        width = 10.0,
        height = 10.0,
        plane = "xy",
        position = (0.0, 0.0, 0.0),
        cell_color = (200, 200, 200),
        section_color = (140, 140, 140)
        )

    urdf = yourdfpy.URDF.load("assets/panda/panda_spherized.urdf")

    panda = ViserUrdf(
        server, urdf, load_meshes = True, load_collision_meshes = False, root_node_name = "/panda"
        )

    panda.update_cfg(
        np.array([-0.04465612, -0.50431913, 0.02652899, -1.93450534, 0.02332041, 1.43755722, 0.77754092])
        )

    control_ee(server, panda)

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
