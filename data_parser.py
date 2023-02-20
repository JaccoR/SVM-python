#%%
import numpy as np
import pandas as pd
from SVM import SVM


#%%
####### GET RANDOM SEPERABLE DATAPOINTS #######

# Set seed so points dont change every run
np.random.seed(1)

# Define the number of data points and their range
n = 1000

# Define limits such that they are perfectly seperable
x_min1, x_max1 = 0, 10
y_min1, y_max1 = 0, 10

x_min2, x_max2 = 0, 10
y_min2, y_max2 = 11, 20

# Generate random data points that are separable
class_1=pd.DataFrame(data={"Feature 1":np.random.uniform(x_min1, x_max1, n//2),"Feature 2":np.random.uniform(y_min1, y_max1, n//2),"label":1})
class_2=pd.DataFrame(data={"Feature 1":np.random.uniform(x_min2, x_max2, n//2),"Feature 2":np.random.uniform(y_min2, y_max2, n//2),"label":-1})

random_points = pd.concat([class_1,class_2], ignore_index=True)

#%%
####### GET MATERIAL DATA #######

material_data = pd.read_csv("DataERC1.txt.txt", sep="   ", engine="python")

# Remove spaces from column names
material_data.columns = material_data.columns.to_series().apply(lambda x: x.strip())

# Set all values in label that are equal to 2 to -1
material_data.loc[material_data['label'] == 2, 'label'] = -1

#%%
####### INITIATE SVM, SOLVE AND VISUALIZE #######

# Random Data
random_points_svm = SVM(random_points)
random_points_svm.train(fraction=0.25)
random_points_svm.predict()
random_points_svm.visualize()

#%%
# Material Data
# material_svm = SVM(material_data)
# material_svm.train(fraction=0.25)
# material_svm.predict()
# material_svm.visualize()




#%%
