from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.CaseRunClass import CaseRun


class Case:
    def __init__(self,inputs=None,outputs=None,ttss_list=None, name = "", input_scaling=None):
        self.inputs = inputs
        self.outputs = outputs
        self.ttss_list = ttss_list
        self.caseruns = [CaseRun(self.inputs,self.outputs,ttss) for ttss in self.ttss_list]
        self.name = name
        self.input_scaling = dict(input_scaling or {})


    def runcases(self):
        for caserun in self.caseruns:
            caserun.name = self.name + " TTSS "+str(caserun.ttss)
            print("Running Case",caserun.name)
            caserun.run_case()

    def plot_overlaid(self):
        if not self.caseruns:
            return

        for case in self.caseruns:
            if case.Y is None or case.t is None:
                case.run_case()

        first = self.caseruns[0]

        n_inputs  = first.sys_id.B.shape[1]
        n_outputs = first.sys_id.C.shape[0]
        input_names  = first.inputs[0]
        output_names = first.outputs[0]

        if len(input_names) != n_inputs or len(output_names) != n_outputs:
            raise ValueError("Input/output names mismatch")
        for case in self.caseruns[1:]:
            if (case.sys_id.B.shape[1] != n_inputs) or (case.sys_id.C.shape[0] != n_outputs):
                raise ValueError("All CaseRun objects must have the same I/O dimensions to overlay.")

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, axes = plt.subplots(
            n_inputs, n_outputs,
            figsize=(4 * n_outputs, 3 * n_inputs),
            squeeze=False, sharex=True
        )

        for j in range(n_inputs):
            for i in range(n_outputs):
                ax = axes[j, i]
                for k, case in enumerate(self.caseruns):
                    ax.plot(case.t, case.Y[(i, j)],
                            linewidth=2,
                            color=colors[k % len(colors)],
                            label=f"TTSS={case.ttss}",
                            alpha = 0.3)
                ax.grid(True, alpha=0.3)

        for i, name in enumerate(output_names):
            axes[0, i].set_title(f"{name}", pad=10)

        for j, name in enumerate(input_names):
            pos = axes[j, 0].get_position()
            fig.text(pos.x0 - 0.04,
                     pos.y0 + pos.height / 2,
                     f"{name}",
                     va='center', ha='right',
                     rotation=90, fontsize=12, fontweight='bold')  #0.02

        legend_handles = [
            Line2D([], [], color=colors[k % len(colors)], linewidth=2, label=f"TTSS={case.ttss}")
            for k, case in enumerate(self.caseruns)
        ]
        fig.legend(handles=legend_handles,
                   loc='upper center', ncol=len(legend_handles),
                   frameon=False)

        plt.show()

    def plot_overlaid_scaled(self):
        #scaled plotting
        if not self.caseruns:
            return

        for case in self.caseruns:
            if case.Y is None or case.t is None:
                case.run_case()

        first = self.caseruns[0]
        n_inputs  = first.sys_id.B.shape[1]
        n_outputs = first.sys_id.C.shape[0]
        input_names  = first.inputs[0]
        output_names = first.outputs[0]

        if len(input_names) != n_inputs or len(output_names) != n_outputs:
            raise ValueError("Input/output names mismatch")
        for case in self.caseruns[1:]:
            if (case.sys_id.B.shape[1] != n_inputs) or (case.sys_id.C.shape[0] != n_outputs):
                raise ValueError("All CaseRun objects must have the same I/O dimensions to overlay.")

        # build aligned per-row scales
        scales = [float(self.input_scaling.get(name, 1.0)) for name in input_names]

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, axes = plt.subplots(n_inputs, n_outputs, figsize=(4 * n_outputs, 3 * n_inputs),
                                 squeeze=False, sharex=True, sharey='col')

        for j in range(n_inputs):
            s = scales[j]
            for i in range(n_outputs):
                ax = axes[j, i]
                for k, case in enumerate(self.caseruns):
                    y = case.Y[(i, j)]
                    ax.plot(case.t, s * y,
                            linewidth=2,
                            color=colors[k % len(colors)],
                            label=f"TTSS={case.ttss}",
                            alpha=0.3)
                ax.grid(True, alpha=0.3)

        # column headers
        for i, name in enumerate(output_names):
            axes[0, i].set_title(f"{name}", pad=10)

        # cow headers
        for j, name in enumerate(input_names):
            tag = f"{name}" if scales[j] == 1.0 else f"{name} × {scales[j]:g}"
            pos = axes[j, 0].get_position()
            fig.text(pos.x0 - 0.04,
                     pos.y0 + pos.height / 2,
                     tag,
                     va='center', ha='right',
                     rotation=90, fontsize=12, fontweight='bold')

        legend_handles = [
            Line2D([], [], color=colors[k % len(colors)], linewidth=2, label=f"TTSS={case.ttss}")
            for k, case in enumerate(self.caseruns)
        ]
        fig.legend(handles=legend_handles, loc='upper center', ncol=len(legend_handles), frameon=False)
        plt.show()




