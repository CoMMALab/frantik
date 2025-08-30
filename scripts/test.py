import time
import numpy as np
from scipy.spatial.transform import Rotation as R

import yourdfpy
import viser
from viser.extras import ViserUrdf

import frantik


def se3_from_pos_xyzw(position, quat_xyzw):
    se3 = np.eye(4)
    se3[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    se3[:3, 3] = np.asarray(position)
    return se3


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

    axes = server.scene.add_frame(name = "/target", axes_length = 0.1)

    urdf = yourdfpy.URDF.load("assets/panda/panda_spherized.urdf")

    panda = ViserUrdf(
        server, urdf, load_meshes = True, load_collision_meshes = False, root_node_name = "/panda"
        )

    global curr_config
    curr_config = np.array(
        [-0.04465612, -0.50431913, 0.02652899, -1.93450534, 0.02332041, 1.43755722, 0.77754092]
        )
    panda.update_cfg(curr_config)

    q_slide = server.gui.add_slider("Q7", min = -175, max = 175, step = 0.01, initial_value = 0)
    x_slide = server.gui.add_slider("X", min = 0, max = 1.0, step = 0.01, initial_value = 0.5)
    y_slide = server.gui.add_slider("Y", min = -1.0, max = 1.0, step = 0.01, initial_value = 0.0)
    z_slide = server.gui.add_slider("Z", min = 0, max = 1.0, step = 0.01, initial_value = 0.5)
    r_slide = server.gui.add_slider("Roll", min = -180, max = 180, step = 1, initial_value = 0)
    t_slide = server.gui.add_slider("Pitch", min = -360, max = 0, step = 1, initial_value = -180)
    p_slide = server.gui.add_slider("Yaw", min = -360, max = 0, step = 1, initial_value = -180)

    def solve_ik():
        global curr_config
        position = [x_slide.value, y_slide.value, z_slide.value]
        rpy = [r_slide.value, t_slide.value, p_slide.value]

        xyzw = R.from_euler('XYZ', rpy, True).as_quat()

        axes.position = np.array(position)
        axes.wxyz = R.from_euler('XYZ', rpy, True).as_quat(scalar_first = True)

        se3 = se3_from_pos_xyzw(position, xyzw)
        input_mat = se3.flatten(order = 'F').tolist()
        config = frantik.cc_ik(input_mat, np.radians(q_slide.value), curr_config.tolist())

        if all(map(lambda c: c == c, config)):
            c = np.array(config)
            panda.update_cfg(c)
            curr_config = c

    sliders = [q_slide, x_slide, y_slide, z_slide, r_slide, t_slide, p_slide]
    for slider in sliders:
        slider.on_update(lambda _: solve_ik())

    solve_ik()

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
