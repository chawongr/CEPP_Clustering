from sklearn.manifold import MDS
import numpy as np

data = np.array([
    [1, 2, 3],  
    [4, 5, 6],   
    [7, 8, 9]    
])

mds = MDS(n_components=2)

# Fit the data to MDS model and transform it to 2D
transformed_data = mds.fit_transform(data)

print("Transformed Data (2D):")
print(transformed_data)
