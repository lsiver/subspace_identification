from Subspace_ID import subspace_id, unit_step_responses, plot_unit_step_responses

class CaseRun:
    inputs = []
    outputs = []
    ttss = 0
    order = 0
    sys_id = None
    transfer_functions = None #dictionary of per channel transfer functions and gains
    t = 0 #time vector for ttss
    Y = None #Simulated unit-step responses


    def __init__(self, inputs=None, outputs=None,ttss=0):
        if outputs is None:
            outputs = []
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = outputs
        self.ttss = int(ttss)

    def run_case(self):
        self.sys_id, self.transfer_functions = subspace_id(self.inputs[1],self.outputs[1])
        self.t, self.Y = unit_step_responses(self.sys_id, self.ttss, step_at = 1)

    def plot_unit_responses(self):
        plot_unit_step_responses(self.t, self.Y, self.sys_id.B.shape[1], self.sys_id.C.shape[0], self.inputs[0],self.outputs[0])




