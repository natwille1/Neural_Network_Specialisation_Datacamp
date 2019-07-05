# Plotting models
# In addition to summarizing your model, you can also plot your model to get a more intuitive sense of it. Your model is available in the workspace.

# Imports
import matplotlib.pyplot as plt
from keras.utils import plot_model
import matplotlib.image as img

# Plot the model
plot_model(model, to_file='model.png')

# Display the image
data = img.imread('model.png')
plt.imshow(data)
plt.show()
