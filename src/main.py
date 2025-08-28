import numpy as np
from identification import identification
import matplotlib.pyplot as plt

if __name__ == "__main__":
    SP = np.loadtxt('SP')
    PV = np.loadtxt('PV')

    inputs = []
    inputs.append(SP)
    outputs = []
    outputs.append(PV)
    vecs =[SP,PV]

    inputsIdx = len(inputs)-1


    #reshape as needed
    for i, vec in enumerate(vecs):
        if vec.ndim == 1:
            vecs[i] = vec.reshape(1,-1)
        else:
            vecs[i] = vec.T

    for i, vec in enumerate(vecs):
        vec_mean = np.mean(vec)
        vecs[i] = vec - vec_mean

    inputs = vecs[:inputsIdx+1]
    outputs = vecs[inputsIdx+1:]

    yid = identification(inputs,outputs)

    npts = vecs[0].shape[1]
    ts = 1.0
    Time = np.arange(0,npts)*ts

    plt.plot(Time, outputs[0].flatten(), 'b-', label="outputs",alpha=0.7)
    plt.plot(Time, yid.flatten())
    plt.legend()
    plt.show()


