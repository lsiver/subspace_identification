import numpy as np

#added src to subspace_ID
from src.Subspace_ID import subspace_id, unit_step_responses, plot_unit_step_responses, subspace_id_transient
from src.MIMO_prediction import MIMOPredictor


class CaseRun:
    def __init__(self, inputs=None, outputs=None,ttss=0):
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.ttss = int(ttss)
        self.order = 0
        self.sys_id = None
        self.transfer_functions = None #dictionary of per channel transfer functions and gains
        self.t = None #time vector for ttss
        self.Y = None #Simulated unit-step responses
        self.name = ""
        self.pred = None


    def run_case(self):
        #self.sys_id, self.transfer_functions = subspace_id(self.inputs[1],self.outputs[1])
        #Shifting the outputs down by 1 generally helps for the very-fast dynamics
        #and you do not lose much doing this...
        shift_samples = 1
        shifted_outputs = np.zeros_like(self.outputs[1])
        shifted_outputs[:, shift_samples:] = self.outputs[1][:, :-shift_samples]
        shifted_outputs[:, :shift_samples] = self.outputs[1][:, 0:1]  # Repeat first value
        self.sys_id, self.transfer_functions = subspace_id_transient(
            self.inputs[1], shifted_outputs, ttss_minutes = self.ttss, k= 2.0
        )
        self.t, self.Y = unit_step_responses(self.sys_id, self.ttss, step_at = 1)

    def plot_unit_responses(self):
        plot_unit_step_responses(self.t, self.Y, self.sys_id.B.shape[1], self.sys_id.C.shape[0], self.inputs[0],self.outputs[0])

    def create_predictor(self):
        self.pred = MIMOPredictor(self)
        return self.pred

    def create_predictions(self):
        if self.pred is None:
            self.create_predictor()

        input_data = self.inputs[1]
        actual_outputs = self.outputs[1]

        try:
            t, y_pred, input_used = self.pred.predict_sequence(input_data)
            self.pred.plot_prediction(t, y_pred, actual_outputs, title="pred vs actual")

        except Exception as e:
            print(f"Error occurred: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()



