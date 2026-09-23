from scipy import sparse as sp
from scipy.sparse import linalg as sla
import numpy as np
import matplotlib.pyplot as plt
AD=np.eye(500) #identity matrix 500x500
A=sp.csr_matrix(AD)
B= sp.diags([np.ones(500),2*np.ones(499) ], [0,1], format="csr")
C=sp.diags([np.ones(500),1/2*np.ones(499),1/2*np.ones(499) ], [0,1,-1], format="csr")
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

for a, M, name in zip(ax, [A, B, C], ["A", "B", "C"]):
    a.spy(M, markersize=2)
    a.set_title(name)
    a.set_xlim(0, 20)     # zoom ind på hjørnet
    a.set_ylim(20, 0)

plt.tight_layout()
plt.show()