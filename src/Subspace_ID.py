from sippy_unipi import system_identification
import matplotlib.pyplot as plt
import numpy as np
from sippy_unipi import functionset as fset
from sippy_unipi import functionsetSIM as fsetSIM
from scipy import signal
from scipy.linalg import hankel

def subspace_id(inputs, outputs,order = None):
    # Identify MIMO system and extract individual transfer functions
    # inputs: array of shape (n_inputs, n_samples)
    # outputs: array of shape (n_outputs, n_samples)
    order = int(determine_optimal_order(inputs, outputs))
    # identify the MIMO system
    sys_id = system_identification(
        outputs,
        inputs,
        "N4SID",
        SS_fixed_order=order
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

def plot_scaled_step_responses(
        t, Y, n_inputs, n_outputs, inputs_names, outputs_names, input_scaling
):

    scales = []
    for name in inputs_names:
        try:
            scales.append(float(input_scaling.get(name, 1.0)))
        except (TypeError, ValueError):
            scales.append(1.0)

    fig, axes = plt.subplots(
        n_inputs, n_outputs,
        figsize=(4 * n_outputs, 3 * n_inputs),
        squeeze=False, sharex=True, sharey='col'
    )

    for j in range(n_inputs):
        s = scales[j]
        for i in range(n_outputs):
            ax = axes[j, i]
            y = np.asarray(Y[(i, j)])
            ax.plot(t, s * y, linewidth=2)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('')
            ax.set_ylabel('')

    # column headers
    for i, output_name in enumerate(outputs_names):
        axes[0, i].set_title(f'{output_name}', pad=10)

    # row headers
    for j, input_name in enumerate(inputs_names):
        tag = f"{input_name}" if scales[j] == 1 else f"{input_name} × {scales[j]:g}"
        pos = axes[j, 0].get_position()
        fig.text(
            pos.x0 - 0.02,
            pos.y0 + pos.height / 2,
            tag,
            va='center', ha='right',
            rotation=90, fontsize=12, fontweight='bold'
        )

    plt.show()

def determine_optimal_order(inputs, outputs, max_order=30):
    # order selection using SVD of Hankel matrix

    n_samples = inputs.shape[1]
    horizon = min(n_samples // 3, 100)

    data = np.vstack([outputs, inputs])

    H = np.zeros((data.shape[0] * horizon, n_samples - horizon))
    for i in range(data.shape[0]):
        for j in range(horizon):
            H[i*horizon + j, :] = data[i, j:j+n_samples-horizon]

    U, s, Vt = np.linalg.svd(H, full_matrices=False)

    # find elbow in singular values
    s_normalized = s / s[0]
    threshold = 0.01
    effective_rank = np.sum(s_normalized > threshold)

    suggested_order = max(2, min(effective_rank // (inputs.shape[0] + outputs.shape[0]), max_order))

    return suggested_order


def subspace_id2(inputs, outputs, order=None):
    U_mean = inputs.mean(axis=1, keepdims=True)
    Y_mean = outputs.mean(axis=1, keepdims=True)
    U_std = inputs.std(axis=1, keepdims=True) + 1e-10
    Y_std = outputs.std(axis=1, keepdims=True) + 1e-10

    U_scaled = (inputs - U_mean) / U_std
    Y_scaled = (outputs - Y_mean) / Y_std

    if order is None:
        order = int(determine_optimal_order(U_scaled, Y_scaled))
        print(f"Auto-selected order: {order}")

    sys_id = system_identification(
        Y_scaled,
        U_scaled,
        "N4SID",
        SS_fixed_order=order,
        SS_D_required=True,
        SS_A_stability=True
    )

    n_inputs = inputs.shape[0]
    n_outputs = outputs.shape[0]
    transfer_functions = {}

    for out_idx in range(n_outputs):
        for in_idx in range(n_inputs):
            C_siso = sys_id.C[out_idx:out_idx+1, :]
            B_siso = sys_id.B[:, in_idx:in_idx+1]
            D_siso = sys_id.D[out_idx:out_idx+1, in_idx:in_idx+1]

            # xfer to transfer fxn
            num, den = signal.ss2tf(sys_id.A, B_siso, C_siso, D_siso)
            num = num[0]

            # calculate actual gain
            dc_gain_scaled = calculate_dc_gain_ss(sys_id.A, B_siso, C_siso, D_siso)
            actual_dc_gain = dc_gain_scaled * (Y_std[out_idx] / U_std[in_idx])

            transfer_functions[(out_idx, in_idx)] = {
                "num": np.asarray(num).ravel(),
                "den": np.asarray(den).ravel(),
                "dc_gain": float(actual_dc_gain),
                "scaled_sys": (sys_id.A, B_siso, C_siso, D_siso),
                "scaling": (U_std[in_idx], Y_std[out_idx]),
                "Ts": 1.0
            }

    return sys_id, transfer_functions

def subspace_id_transient(U, Y, ttss_minutes, order=None, k=2.0):
    if order is None:
        order = int(determine_optimal_order(U, Y))

    change_times = detect_changes_adaptive(U, q=0.9, min_gap=30)

    # Extract transient windows
    Uc, Yc = make_transient_dataset(U, Y, change_times, ttss_minutes, k=k, pre=10, spacer=2)

    if Uc is None:
        Uc, Yc = U, Y

    return subspace_id(Uc, Yc, order=order)

def detect_changes_adaptive(U, q=0.9, min_gap=30):
    # detect when step changes occur in the input signals
    m, N = U.shape

    U_norm = np.zeros_like(U)
    for i in range(m):
        mean = U[i, :].mean()
        std = U[i, :].std() + 1e-10
        U_norm[i, :] = (U[i, :] - mean) / std

    dU = np.abs(np.diff(U_norm, axis=1, prepend=U_norm[:, :1]))

    thresholds = np.zeros(m)
    for i in range(m):
        thresholds[i] = np.quantile(dU[i, :], q)

    change_times = []
    last_change = -min_gap - 1

    for t in range(1, N):
        if np.any(dU[:, t] > thresholds):
            if t - last_change >= min_gap:
                change_times.append(t)
                last_change = t

    return change_times

def make_transient_dataset(U, Y, change_times, ttss_min, k=2.0, pre=10, spacer=2):
    # extract windows around step changes
    if len(change_times) == 0:
        return None, None

    n_in, N = U.shape
    n_out = Y.shape[0]

    window_size = int(np.ceil(k * ttss_min))

    Uc_segments = []
    Yc_segments = []

    for t0 in change_times:
        t_start = max(0, t0 - pre)
        t_end = min(N, t0 + window_size)

        if (t_end - t_start) < (pre + ttss_min//2):
            continue

        u_seg = U[:, t_start:t_end].copy().astype(np.float64)
        y_seg = Y[:, t_start:t_end].copy().astype(np.float64)

        baseline_start = max(0, t0 - pre)
        baseline_end = t0

        if baseline_end > baseline_start:
            u_baseline = U[:, baseline_start:baseline_end].mean(axis=1, keepdims=True)
            y_baseline = Y[:, baseline_start:baseline_end].mean(axis=1, keepdims=True)

            u_seg -= u_baseline
            y_seg -= y_baseline

        if t0 - t_start > 0:
            u_seg[:, :t0-t_start] = 0.0


        Uc_segments.append(u_seg)
        Yc_segments.append(y_seg)

        if spacer > 0:
            Uc_segments.append(np.zeros((n_in, spacer)))
            Yc_segments.append(np.zeros((n_out, spacer)))

    if len(Uc_segments) == 0:
        return None, None

    Uc = np.concatenate(Uc_segments, axis=1)
    Yc = np.concatenate(Yc_segments, axis=1)

    return Uc, Yc
