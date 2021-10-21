import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('crop_yield')

data = data.transpose()
rainfall = data[0]
crop_yield = data[1]

plt.plot(rainfall, crop_yield, 'b.', label='Annual Crop Yield by Rainfall')
plt.xlabel('Annual Rainfaill (mm)')
plt.ylabel('Annual Crop Yield (tonnes/hectare)')
plt.legend()
plt.show()

### Part 1
### TODO: Create a function that returns a design matrix of degree d from data x
def phiMat():
    phi = None
    return phi

def fit_polynomial(data_x,data_y,degree):
    ### TODO: pass the correct arguments into the phiMat function
    Phi = phiMat()
    A = Phi.T@Phi
    B = Phi.T@data_y
    return np.linalg.solve(A,B)

### TODO: Fit a degree 1 polynomial and plot the observed data and the model


### Part 2: Verify the model using sklearn linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# reshape the data to fit the shape accepted by sklearn
X = rainfall.reshape(len(rainfall),1)
# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, crop_yield, test_size = 0.25, random_state=0)

### TODO: Plot the observed data as points, 
#         the sklearn model as a line, 
#         and the model's prediction of the test data as points

# hint: use .intercept_ and .coef_ for plotting the model

#Example for plotting a function over every 10th x value from 1 to 100:
#x = np.linspace(1,100,10)
#y = 2*x + 30
#z = 3*x - 20
#plt.plot(x, y, 'b', label='2*x+1')
#plt.plot(x, z, 'y', label='3*x-20')
#plt.legend()
#plt.show()
