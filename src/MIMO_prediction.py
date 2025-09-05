import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class MIMOPredictor:
    def __init__(self, caserun):
        if caserun.sys_id is None or caserun.transfer_functions is None:
            raise ValueError("CaseRun must be trained first (call run_case())")

        self.caserun = caserun
        self.sys_id = caserun.sys_id
        self.transfer_functions = caserun.transfer_functions
        self.n_inputs = self.sys_id.B.shape[1]
        self.n_outputs = self.sys_id.C.shape[0]
        self.Ts = 1.0  # Sampling time not likely to change

        self.reset_state()

    def reset_state(self):
        self.x = np.zeros((self.sys_id.A.shape[0], 1))
        self.last_inputs = np.zeros((self.n_inputs, 1))
        self.prediction_history = {
            'time': [],
            'inputs': [],
            'outputs': [],
            'states': []
        }

    def predict_single_step(self, current_inputs):
        if current_inputs.ndim == 1:
            current_inputs = current_inputs.reshape(-1, 1)

        # state update x[k+1] = A*x[k] + B*u[k]
        self.x = self.sys_id.A @ self.x + self.sys_id.B @ current_inputs

        # output y[k] = C*x[k] + D*u[k]
        outputs = self.sys_id.C @ self.x + self.sys_id.D @ current_inputs

        return outputs.flatten()

    def predict_sequence(self, input_sequence, initial_state=None):
        # input format
        if input_sequence.ndim == 2 and input_sequence.shape[0] == self.n_inputs:
            # (n_inputs, n_samples) -> (n_samples, n_inputs)
            input_sequence = input_sequence.T

        n_samples = input_sequence.shape[0]

        if initial_state is not None:
            self.x = initial_state.reshape(-1, 1)
        else:
            self.reset_state()

        outputs = np.zeros((n_samples, self.n_outputs))

        for k in range(n_samples):
            current_input = input_sequence[k, :].reshape(-1, 1)
            outputs[k, :] = self.predict_single_step(current_input)

        time_vector = np.arange(n_samples) * self.Ts

        return time_vector, outputs, input_sequence

    def create_step_input_sequence(self, step_times, step_values, total_time, input_channel=0):
        n_samples = int(total_time)
        input_sequence = np.zeros((n_samples, self.n_inputs))

        current_value = 0.0

        for i, (step_time, step_value) in enumerate(zip(step_times, step_values)):
            step_sample = int(step_time)
            if step_sample < n_samples:
                step_change = step_value - current_value
                input_sequence[step_sample:, input_channel] += step_change
                current_value = step_value

        return input_sequence

    def predict_from_steps(self, step_times, step_values, total_time, input_channel=0):
        input_sequence = self.create_step_input_sequence(
            step_times, step_values, total_time, input_channel
        )

        return self.predict_sequence(input_sequence)

    def plot_prediction(self, time_vector, outputs, actual_outputs, title="MIMO Prediction"):
        if len(self.caserun.outputs) > 0:
            output_names = self.caserun.outputs[0]
        else:
            output_names = []
            for i in range(self.n_outputs):
                output_names.append(f"Output {i}")

        #separate fig for each output. Not the greatest way to do this. Want a different plotting library
        #later improvement.
        for i in range(self.n_outputs):
            plt.figure(figsize=(12, 6))

            #calculate avgs so i can shift the data to match so you do not need to
            #scale axes separately
            skip_samples = int(0.05*len(outputs))
            predicted_avg = np.mean(outputs[skip_samples:, i])

            actual_length = min(len(time_vector), actual_outputs.shape[1])
            actual_avg = np.mean(actual_outputs[i, skip_samples:actual_length])

            offset = actual_avg - predicted_avg

            shifted_outputs = outputs[:, i] + offset

            plt.plot(time_vector, shifted_outputs, label=f"Predicted {output_names[i]}",
                     linewidth=2, linestyle='-', color='blue')

            actual_length = min(len(time_vector), actual_outputs.shape[1])
            plt.plot(time_vector[:actual_length], actual_outputs[i, :actual_length],
                     label=f"Actual {output_names[i]}", linewidth=2, linestyle='-',
                     color='red', alpha=0.5)

            plt.title(f"{title} - {output_names[i]}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

        plt.show()