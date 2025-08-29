import numpy as np
from identification import identification
import matplotlib.pyplot as plt

if __name__ == "__main__":
    SP = np.loadtxt('SP').reshape(1,-1)
    PV = np.loadtxt('PV').reshape(1,-1)

    # inputs = []
    # inputs.append(SP)
    # outputs = []
    # outputs.append(PV)
    # vecs =[SP,PV]
    #
    # inputsIdx = len(inputs)-1
    #
    #
    # #reshape as needed
    # for i, vec in enumerate(vecs):
    #     if vec.ndim == 1:
    #         vecs[i] = vec.reshape(1,-1)
    #     else:
    #         vecs[i] = vec.T
    #
    # for i, vec in enumerate(vecs):
    #     vec_mean = np.mean(vec)
    #     vecs[i] = vec - vec_mean
    #
    # inputs = np.array(vecs[:inputsIdx+1])
    # outputs = np.array(vecs[inputsIdx+1:])
    # print(inputs)

    # yid = identification(inputs,outputs)
    #
    # npts = vecs[0].shape[1]
    # ts = 1.0
    # Time = np.arange(0,npts)*ts
    #
    # plt.plot(Time, outputs[0].flatten(), 'b-', label="outputs",alpha=0.7)
    # plt.plot(Time, yid.flatten())
    # plt.legend()
    # plt.show()

    # from main2 import mimo_identification_and_analysis
    # sys_id, transfer_functions, responses = mimo_identification_and_analysis(
    #     SP, PV, system_order=2
    # )
    # Example: 2 inputs, 3 outputs
    n_samples = 500
    n_inputs = 2
    n_outputs = 3

    # # Generate example data (replace with your actual data)
    inputs = np.random.randn(n_inputs, n_samples)
    outputs = np.random.randn(n_outputs, n_samples)
    print(inputs.shape)
    from main3 import mimo_identification_and_analysis
    sys_id, transfer_functions, responses = mimo_identification_and_analysis(
        inputs, outputs, system_order=3
    )

    # print(inputs.shape)

