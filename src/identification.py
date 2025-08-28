from sippy_unipi import system_identification
import matplotlib.pyplot as plt
import numpy as np
from sippy_unipi import functionset as fset
from sippy_unipi import functionsetSIM as fsetSIM

def identification(inputs,outputs):

    #will need to optimize this eventually
    system_order = 2
    sys_id = system_identification(
        outputs[0],
        inputs[0],
        "N4SID",
        SS_fixed_order=system_order
    )
    xid, yid = fsetSIM.SS_lsim_process_form(
        sys_id.A,
        sys_id.B,
        sys_id.C,
        sys_id.D,
        inputs[0],
        sys_id.x0
    )

    return yid

   #  plt.plot(Time, outputs.flatten(), 'b-', label="outputs",alpha=0.7)
   # # plt.plot(Time, inputs.flatten(), 'g-',label="inputs")
   #  plt.plot(Time, yid.flatten())
   #  plt.legend()
   #  plt.show()


