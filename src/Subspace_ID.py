from sippy_unipi import system_identification
import matplotlib.pyplot as plt
import numpy as np
from sippy_unipi import functionset as fset
from sippy_unipi import functionsetSIM as fsetSIM
from scipy import signal

def subspace_id(inputs, outputs):
    # Identify MIMO system and extract individual transfer functions
    # inputs: array of shape (n_inputs, n_samples)
    # outputs: array of shape (n_outputs, n_samples)
    #System order should be >= num of outputs
    #Will optimize this later...
    system_order = max(2, outputs.shape[0])
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
            C_siso = sys_id.C[out_idx:out_idx+1, :]  # output row
            B_siso = sys_id.B[:, in_idx:in_idx+1]     # input column
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

def subspace_id_transient(U, Y, ttss_minutes, order=None, k=2.0):
    if order is None:
        order = max(2, Y.shape[0])
    change_times = detect_changes_adaptive(U, q=0.9, min_gap=30)
    Uc, Yc = make_transient_dataset(U, Y, change_times, ttss_minutes, k=k, pre=10, spacer=2)
    if Uc is None:
        Uc, Yc = U, Y
    return subspace_id(Uc, Yc)

def calculate_dc_gain_ss(A, B, C, D):
    #gain C (I - A)^(-1) B + D
    I = np.eye(A.shape[0])
    try:
        X = np.linalg.solve(I - A, B)
        return float(C @ X + D)
    except np.linalg.LinAlgError:
        #? pole at z = 1 (integrator, level, open pressure). No steady-state gain
        return np.nan



def unit_step_responses(sys_id, n_steps=400, step_at=1):
    # Simulate MIMO responses to a +1-unit step on each input (others = 0).
    # Outputs stay in their native units.
    #  t ~ (n_steps,) time vector in seconds
    #  Y ~ dict[(out_idx, in_idx)] = (n_steps,) response yi to unit step in uj
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

def detect_changes_adaptive(U, q=0.9, min_gap=30):
    Um = U.mean(axis=1, keepdims=True)
    Us = U.std(axis=1, keepdims=True) + 1e-9
    UN = (U - Um) / Us
    dUN = np.abs(np.diff(UN, axis=1, prepend=UN[:, :1]))
    thr = np.quantile(dUN, q)
    m, N = UN.shape
    last = np.full(m, -min_gap-1, dtype=int)
    times = set()
    for t in range(1, N):
        fired = False
        for j in range(m):
            if dUN[j, t] >= thr and (t - last[j]) >= min_gap:
                last[j] = t
                fired = True
        if fired:
            times.add(t)
    return sorted(times)

def make_transient_dataset(U, Y, change_times, ttss_min, k=2.0, pre=10, spacer=2):
    n_in, N = U.shape
    n_out = Y.shape[0]
    win = int(np.ceil(k * ttss_min))
    Uc, Yc = [], []
    for t0 in change_times:
        t_start = max(0, t0 - pre)
        t_end = min(N, t0 + win)
        if (t_end - t_start) < (pre + 5):
            continue
        u_seg = U[:, t_start:t_end].copy()
        y_seg = Y[:, t_start:t_end].copy()
        u_base = U[:, max(0, t0-pre):t0].mean(axis=1, keepdims=True)
        y_base = Y[:, max(0, t0-pre):t0].mean(axis=1, keepdims=True)
        u_seg -= u_base
        y_seg -= y_base
        u_seg[:, :min(pre, u_seg.shape[1])] = 0.0
        Uc.append(u_seg)
        Yc.append(y_seg)
        if spacer>0:
            Uc.append(np.zeros((n_in, spacer)))
            Yc.append(np.zeros((n_out, spacer)))
    if not Uc:
        return None, None
    return np.concatenate(Uc, axis=1), np.concatenate(Yc, axis=1)

if __name__ == "__main__":
    n_samples = 500
    n_inputs = 2
    n_outputs = 3
