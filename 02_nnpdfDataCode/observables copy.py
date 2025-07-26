import numpy as np
from matplotlib import pyplot as plt

# If running on separate laptop/computer, this will need commenting out 
plt.style.use('pythonStyle')
import pythonStyle as ed

matrix = np.loadtxt('correlation_kdeEDIT.csv', delimiter=',')

# Plot correlation matrix
plt.figure(figsize=(10, 8))
im_corr = plt.imshow(matrix, aspect='auto')
cbar_corr = plt.colorbar(im_corr)
cbar_corr.set_label("Correlation")
plt.gca().invert_yaxis()
plt.grid(False)
plt.savefig('correlationMatrixFull.png')
plt.show()