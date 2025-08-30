from sippy_unipi import system_identification
import matplotlib.pyplot as plt
import numpy as np
from sippy_unipi import functionset as fset
from sippy_unipi import functionsetSIM as fsetSIM
from scipy import signal

def subspace_id(inputs, outputs):
    """
    Identify MIMO system and extract individual transfer functions
    inputs: array of shape (n_inputs, n_samples)
    outputs: array of shape (n_outputs, n_samples)
    """
    #System order should be >= num of outputs
    #Will optimize this later...
    if outputs.shape[0] < 2:
        system_order = 2
    else:
        system_order = outputs.shape[0]

    # identify the MIMO system
    sys_id = system_identification(
        outputs,
        inputs,
        "N4SID",
        SS_fixed_order=system_order
    )

    # get system dimensions
    n_inputs = inputs.shape[0]
    n_outputs = outputs.shape[0]
    n_samples = inputs.shape[1]

    # transfer function matrix
    # each element G[i,j] is the transfer function from input j to output i
    transfer_functions = {}

    ts = 1.0

    for out_idx in range(n_outputs):
        for in_idx in range(n_inputs):
            # extract SISO transfer function from input j to output i
            # Using scipy's state-space to transfer function conversion

            # create SISO system
            C_siso = sys_id.C[out_idx:out_idx+1, :]  # select output row
            B_siso = sys_id.B[:, in_idx:in_idx+1]     # select input column
            D_siso = sys_id.D[out_idx:out_idx+1, in_idx:in_idx+1]

            # xfer to transfer function
            num, den = signal.ss2tf(sys_id.A, B_siso, C_siso, D_siso)
            num = num[0]

            transfer_functions[(out_idx, in_idx)] = {
                "num": np.asarray(num).ravel(),
                "den": np.asarray(den).ravel(),
                "dc_gain": calculate_dc_gain_ss(sys_id.A, B_siso, C_siso, D_siso),
                "C": C_siso, "B": B_siso, "D": D_siso,
                "Ts": ts
            }

    return sys_id, transfer_functions

def calculate_dc_gain_ss(A, B, C, D):
    #gain: C (I - A)^(-1) B + D
    I = np.eye(A.shape[0])
    try:
        X = np.linalg.solve(I - A, B)
        return float(C @ X + D)
    except np.linalg.LinAlgError:
        #? pole at z = 1 (integrator, level, open pressure). No steady-state gain
        return np.nan



def unit_step_responses(sys_id, n_steps=400, step_at=1):
    """
    Simulate MIMO responses to a +1-unit step on each input (others = 0).
    Outputs stay in their native units.
    """
    #  t : (n_steps,) time vector in seconds
    #  Y : dict[(out_idx, in_idx)] -> (n_steps,) response y_i to unit step in u_j
    Ts = 1.0

    n_out = sys_id.C.shape[0]
    n_in  = sys_id.B.shape[1]

    t = np.arange(n_steps) * Ts
    Y = {}

    #build one input at a time and simulate with full MIMO model
    for j in range(n_in):
        U = np.zeros((n_steps, n_in))
        U[step_at:, j] = 1.0

        #discrete SS
        sysd = signal.dlti(sys_id.A, sys_id.B, sys_id.C, sys_id.D, dt=Ts)

        tout, y, x = signal.dlsim(sysd, U)
        y = np.asarray(y)

        for i in range(n_out):
            Y[(i, j)] = y[:, i]

    return t, Y

def plot_unit_step_responses(t, Y, n_inputs, n_outputs, inputs_names, outputs_names):
    fig, axes = plt.subplots(
        n_inputs, n_outputs,
        figsize=(4 * n_outputs, 3 * n_inputs),
        squeeze=False, sharex=True
    )

    for j in range(n_inputs):
        for i in range(n_outputs):
            ax = axes[j, i]
            ax.plot(t, Y[(i, j)], linewidth=2)
            ax.grid(True, alpha=0.3)

            ax.set_xlabel('')
            ax.set_ylabel('')

    # column headers
    for i, output_name in enumerate(outputs_names):
        axes[0, i].set_title(f'{output_name}', pad=10)

    # row headers
    for j, input_name in enumerate(inputs_names):
        pos = axes[j, 0].get_position()
        fig.text(
            pos.x0 - 0.02,
            pos.y0 + pos.height / 2,
            f'{input_name}',
            va='center', ha='right',
            rotation=90, fontsize=12, fontweight='bold'
        )

    plt.show()

if __name__ == "__main__":
    n_samples = 500
    n_inputs = 2
    n_outputs = 3
