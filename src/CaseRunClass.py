from Subspace_ID import subspace_id, unit_step_responses, plot_unit_step_responses

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


    def run_case(self):
        self.sys_id, self.transfer_functions = subspace_id(self.inputs[1],self.outputs[1])
        self.t, self.Y = unit_step_responses(self.sys_id, self.ttss, step_at = 1)

    def plot_unit_responses(self):
        plot_unit_step_responses(self.t, self.Y, self.sys_id.B.shape[1], self.sys_id.C.shape[0], self.inputs[0],self.outputs[0])




