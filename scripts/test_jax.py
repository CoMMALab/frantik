import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import yourdfpy
import viser
from viser.extras import ViserUrdf

# Import the JAX IK implementation
import jax
import jax.numpy as jnp
from typing import NamedTuple

class FrankaConstants:
    """Constants for Franka Emika Panda robot"""
    # Link parameters
    d1 = 0.3330
    d3 = 0.3160
    d5 = 0.3840
    d7e = 0.2104
    a4 = 0.0825
    a7 = 0.0880
    
    # Precomputed geometric values
    LL24 = 0.10666225  # a4^2 + d3^2
    LL46 = 0.15426225  # a4^2 + d5^2
    L24 = 0.326591870689  # sqrt(LL24)
    L46 = 0.392762332715  # sqrt(LL46)
    
    thetaH46 = 1.35916951803  # atan(d5/a4)
    theta342 = 1.31542071191  # atan(d3/a4)
    theta46H = 0.211626808766  # pi/2 - atan(d5/a4)
    
    # Joint limits
    q_min = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = jnp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    
    NEAR_ONE = 0.999

@jax.jit
def franka_ik(tf: jnp.ndarray, q7: float, q_c: jnp.ndarray) -> jnp.ndarray:
    """
    Analytical inverse kinematics for Franka Emika Panda robot.
    
    Args:
        tf: 4x4 transformation matrix of end-effector pose
        q7: Fixed value for joint 7 (radians)
        q_c: Current joint configuration (7,) - used for singularity handling
    
    Returns:
        Array of shape (4, 7) containing up to 4 IK solutions.
        Invalid solutions are marked with NaN.
    """
    
    # Initialize output with NaN
    q_all = jnp.full((4, 7), jnp.nan)
    
    # Check q7 bounds
    q7_valid = (q7 > FrankaConstants.q_min[6]) & (q7 < FrankaConstants.q_max[6])
    
    def compute_ik():
        # Set q7 for all solutions
        q_all_with_q7 = q_all.at[:, 6].set(q7)
        
        # Extract pose components
        R_EE = tf[:3, :3]
        z_EE = tf[:3, 2]
        p_EE = tf[:3, 3]
        
        # Compute p_6
        p_7 = p_EE - FrankaConstants.d7e * z_EE
        x_EE_6 = jnp.array([jnp.cos(q7 - jnp.pi/4), -jnp.sin(q7 - jnp.pi/4), 0.0])
        x_6 = R_EE @ x_EE_6
        x_6 = x_6 / jnp.linalg.norm(x_6)
        p_6 = p_7 - FrankaConstants.a7 * x_6
        
        # Compute q4
        p_2 = jnp.array([0.0, 0.0, FrankaConstants.d1])
        V26 = p_6 - p_2
        LL26 = jnp.sum(V26**2)
        L26 = jnp.sqrt(LL26)
        
        # Check triangle inequality
        triangle_valid = (
            (FrankaConstants.L24 + FrankaConstants.L46 >= L26) &
            (FrankaConstants.L24 + L26 >= FrankaConstants.L46) &
            (L26 + FrankaConstants.L46 >= FrankaConstants.L24)
        )
        
        theta246 = jnp.arccos(
            (FrankaConstants.LL24 + FrankaConstants.LL46 - LL26) / 
            (2.0 * FrankaConstants.L24 * FrankaConstants.L46)
        )
        q4 = theta246 + FrankaConstants.thetaH46 + FrankaConstants.theta342 - 2.0 * jnp.pi
        
        q4_valid = (q4 > FrankaConstants.q_min[3]) & (q4 < FrankaConstants.q_max[3])
        
        # Set q4 for all solutions
        q_all_with_q4 = q_all_with_q7.at[:, 3].set(q4)
        
        # Compute q6 candidates
        theta462 = jnp.arccos(
            (LL26 + FrankaConstants.LL46 - FrankaConstants.LL24) / 
            (2.0 * L26 * FrankaConstants.L46)
        )
        theta26H = FrankaConstants.theta46H + theta462
        D26 = -L26 * jnp.cos(theta26H)
        
        Z_6 = jnp.cross(z_EE, x_6)
        Y_6 = jnp.cross(Z_6, x_6)
        R_6 = jnp.column_stack([x_6, Y_6 / jnp.linalg.norm(Y_6), Z_6 / jnp.linalg.norm(Z_6)])
        
        V_6_62 = R_6.T @ (-V26)
        Phi6 = jnp.arctan2(V_6_62[1], V_6_62[0])
        Theta6 = jnp.arcsin(D26 / jnp.sqrt(V_6_62[0]**2 + V_6_62[1]**2))
        
        # Two q6 solutions
        q6_candidates = jnp.array([jnp.pi - Theta6 - Phi6, Theta6 - Phi6])
        
        # Wrap q6 to joint limits
        q6_wrapped = jnp.where(
            q6_candidates <= FrankaConstants.q_min[5],
            q6_candidates + 2.0 * jnp.pi,
            jnp.where(
                q6_candidates >= FrankaConstants.q_max[5],
                q6_candidates - 2.0 * jnp.pi,
                q6_candidates
            )
        )
        
        q6_valid = (q6_wrapped > FrankaConstants.q_min[5]) & (q6_wrapped < FrankaConstants.q_max[5])
        
        # Set q6 values (each q6 solution maps to 2 configurations)
        q_all_with_q6 = q_all_with_q4.at[0, 5].set(q6_wrapped[0])
        q_all_with_q6 = q_all_with_q6.at[1, 5].set(q6_wrapped[0])
        q_all_with_q6 = q_all_with_q6.at[2, 5].set(q6_wrapped[1])
        q_all_with_q6 = q_all_with_q6.at[3, 5].set(q6_wrapped[1])
        
        # Compute q1, q2 for each q6 solution
        thetaP26 = 3.0 * jnp.pi/2 - theta462 - theta246 - FrankaConstants.theta342
        thetaP = jnp.pi - thetaP26 - theta26H
        LP6 = L26 * jnp.sin(thetaP26) / jnp.sin(thetaP)
        
        def compute_q1_q2_for_q6(q6_val):
            z_6_5 = jnp.array([jnp.sin(q6_val), jnp.cos(q6_val), 0.0])
            z_5 = R_6 @ z_6_5
            V2P = p_6 - LP6 * z_5 - p_2
            L2P = jnp.linalg.norm(V2P)
            
            # Handle singularity
            is_singular = jnp.abs(V2P[2] / L2P) > FrankaConstants.NEAR_ONE
            
            # Normal case
            q1_normal = jnp.arctan2(V2P[1], V2P[0])
            q2_normal = jnp.arccos(V2P[2] / L2P)
            
            # Two solutions: original and flipped
            q1_solutions = jnp.array([
                q1_normal,
                q1_normal + jnp.where(q1_normal < 0, jnp.pi, -jnp.pi)
            ])
            q2_solutions = jnp.array([q2_normal, -q2_normal])
            
            # Singular case: use current configuration
            q1_solutions = jnp.where(is_singular, q_c[0], q1_solutions)
            q2_solutions = jnp.where(is_singular, 0.0, q2_solutions)
            
            return q1_solutions, q2_solutions, V2P, z_5
        
        # Compute for both q6 values
        q1_sols_0, q2_sols_0, V2P_0, z_5_0 = compute_q1_q2_for_q6(q6_wrapped[0])
        q1_sols_1, q2_sols_1, V2P_1, z_5_1 = compute_q1_q2_for_q6(q6_wrapped[1])
        
        # Set q1, q2 values
        q_all_partial = q_all_with_q6.at[0, 0].set(q1_sols_0[0])
        q_all_partial = q_all_partial.at[0, 1].set(q2_sols_0[0])
        q_all_partial = q_all_partial.at[1, 0].set(q1_sols_0[1])
        q_all_partial = q_all_partial.at[1, 1].set(q2_sols_0[1])
        q_all_partial = q_all_partial.at[2, 0].set(q1_sols_1[0])
        q_all_partial = q_all_partial.at[2, 1].set(q2_sols_1[0])
        q_all_partial = q_all_partial.at[3, 0].set(q1_sols_1[1])
        q_all_partial = q_all_partial.at[3, 1].set(q2_sols_1[1])
        
        # Store V2P and z_5 for q3, q5 computation
        V2P_all = jnp.array([V2P_0, V2P_0, V2P_1, V2P_1])
        z_5_all = jnp.array([z_5_0, z_5_0, z_5_1, z_5_1])
        
        # Compute q3 and q5 for each configuration
        def compute_q3_q5(i):
            q_config = q_all_partial[i]
            V2P_i = V2P_all[i]
            z_5_i = z_5_all[i]
            
            # Check q1, q2 bounds
            q12_valid = (
                (q_config[0] > FrankaConstants.q_min[0]) & 
                (q_config[0] < FrankaConstants.q_max[0]) &
                (q_config[1] > FrankaConstants.q_min[1]) & 
                (q_config[1] < FrankaConstants.q_max[1])
            )
            
            # Compute q3
            z_3 = V2P_i / jnp.linalg.norm(V2P_i)
            Y_3 = -jnp.cross(V26, V2P_i)
            y_3 = Y_3 / jnp.linalg.norm(Y_3)
            x_3 = jnp.cross(y_3, z_3)
            
            c1, s1 = jnp.cos(q_config[0]), jnp.sin(q_config[0])
            R_1 = jnp.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
            
            c2, s2 = jnp.cos(q_config[1]), jnp.sin(q_config[1])
            R_1_2 = jnp.array([[c2, -s2, 0], [0, 0, 1], [-s2, -c2, 0]])
            
            R_2 = R_1 @ R_1_2
            x_2_3 = R_2.T @ x_3
            q3 = jnp.arctan2(x_2_3[2], x_2_3[0])
            
            q3_valid = (q3 > FrankaConstants.q_min[2]) & (q3 < FrankaConstants.q_max[2])
            
            # Compute q5
            VH4 = p_2 + FrankaConstants.d3 * z_3 + FrankaConstants.a4 * x_3 - p_6 + FrankaConstants.d5 * z_5_i
            
            c6, s6 = jnp.cos(q_config[5]), jnp.sin(q_config[5])
            R_5_6 = jnp.array([[c6, -s6, 0], [0, 0, -1], [s6, c6, 0]])
            R_5 = R_6 @ R_5_6.T
            V_5_H4 = R_5.T @ VH4
            
            q5 = -jnp.arctan2(V_5_H4[1], V_5_H4[0])
            q5_valid = (q5 > FrankaConstants.q_min[4]) & (q5 < FrankaConstants.q_max[4])
            
            # Return updated configuration with validity check
            all_valid = q12_valid & q3_valid & q5_valid
            updated_config = q_config.at[2].set(q3).at[4].set(q5)
            return jnp.where(all_valid, updated_config, jnp.full(7, jnp.nan))
        
        # Process all 4 configurations
        final_configs = jax.vmap(compute_q3_q5)(jnp.arange(4))
        
        # Apply validity masks for q6
        q6_mask = jnp.array([q6_valid[0], q6_valid[0], q6_valid[1], q6_valid[1]])
        final_configs = jnp.where(
            q6_mask[:, None], 
            final_configs, 
            jnp.full((4, 7), jnp.nan)
        )
        
        # Apply overall validity
        overall_valid = q7_valid & triangle_valid & q4_valid
        return jnp.where(overall_valid, final_configs, jnp.full((4, 7), jnp.nan))
    
    return jnp.where(q7_valid, compute_ik(), q_all)


# Utility functions
@jax.jit 
def is_valid_solution(q_solution: jnp.ndarray) -> bool:
    """Check if any joint in the solution is finite (valid)"""
    return jnp.any(jnp.isfinite(q_solution[5]))  # Check q6 as indicator

@jax.jit
def count_valid_solutions(q_all: jnp.ndarray) -> int:
    """Count number of valid solutions in the output"""
    return jnp.sum(jnp.isfinite(q_all[:, 5]))

def jax_cc_ik(se3, q7_deg, curr_config):
    """JAX-based equivalent of frantik.cc_ik"""
    # Convert inputs to JAX arrays
    tf = jnp.array(se3)
    q7_rad = jnp.deg2rad(q7_deg)  # Convert degrees to radians
    q_c = jnp.array(curr_config)
    
    # Solve IK - returns (4, 7) array of solutions
    solutions = franka_ik(tf, q7_rad, q_c)
    
    # Find first valid solution
    for i in range(4):
        solution = solutions[i]
        if jnp.isfinite(solution[5]):  # Check if q6 is finite (valid solution indicator)
            return np.array(solution)
    
    # Return NaN array if no valid solution found
    return np.full(7, np.nan)

def is_valid_config(config):
    """Check if configuration is valid (no NaN values)"""
    return np.all(np.isfinite(config))

def se3_from_pos_xyzw(position, quat_xyzw):
    se3 = np.eye(4)
    se3[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    se3[:3, 3] = np.asarray(position)
    return se3

def main():
    server = viser.ViserServer()
    floor_grid = server.scene.add_grid(
        name="/floor_grid",
        width=10.0,
        height=10.0,
        plane="xy",
        position=(0.0, 0.0, 0.0),
        cell_color=(200, 200, 200),
        section_color=(140, 140, 140)
    )
    axes = server.scene.add_frame(name="/target", axes_length=0.1)
    urdf = yourdfpy.URDF.load("assets/panda/panda_spherized.urdf")
    panda = ViserUrdf(
        server, urdf, load_meshes=True, load_collision_meshes=False, root_node_name="/panda"
    )
    
    global curr_config
    curr_config = np.array(
        [-0.04465612, -0.50431913, 0.02652899, -1.93450534, 0.02332041, 1.43755722, 0.77754092]
    )
    panda.update_cfg(curr_config)
    
    # Convert joint limits from radians to degrees for sliders
    q7_min_deg = np.rad2deg(FrankaConstants.q_min[6])
    q7_max_deg = np.rad2deg(FrankaConstants.q_max[6])
    
    q_slide = server.gui.add_slider("Q7", min=q7_min_deg, max=q7_max_deg, step=0.01, initial_value=0)
    x_slide = server.gui.add_slider("X", min=0, max=1.0, step=0.01, initial_value=0.5)
    y_slide = server.gui.add_slider("Y", min=-1.0, max=1.0, step=0.01, initial_value=0.0)
    z_slide = server.gui.add_slider("Z", min=0, max=1.0, step=0.01, initial_value=0.5)
    r_slide = server.gui.add_slider("Roll", min=-180, max=180, step=1, initial_value=0)
    t_slide = server.gui.add_slider("Pitch", min=-360, max=0, step=1, initial_value=-180)
    p_slide = server.gui.add_slider("Yaw", min=-360, max=0, step=1, initial_value=-180)
    
    def solve_ik():
        global curr_config
        position = [x_slide.value, y_slide.value, z_slide.value]
        rpy = [r_slide.value, t_slide.value, p_slide.value]
        xyzw = R.from_euler('XYZ', rpy, True).as_quat()
        axes.position = np.array(position)
        axes.wxyz = R.from_euler('XYZ', rpy, True).as_quat(scalar_first=True)
        se3 = se3_from_pos_xyzw(position, xyzw)
        
        # Replace frantik.cc_ik with JAX implementation
        config = jax_cc_ik(se3, q_slide.value, curr_config)
        
        # Replace validation logic
        if is_valid_config(config):
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
