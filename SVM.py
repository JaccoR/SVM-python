#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

class SVM:
    def __init__(self, data):
        self.data = data
        
        # Convert pandas to numpy array and set the X and y
        self.X = self.data[["Feature 1","Feature 2"]].to_numpy()
        self.y = self.data["label"].to_numpy()

        # Get some convenient values 
        self.n_points, self.n_features = self.X.shape
        self.X_max, self.X_min = self.X[:,0].max(), self.X[:,0].min()

    # Solves the dual problem for a set of data
    def train(self, fraction=1):

        # Select a sample fraction of the dataset
        self.n_samples = int(self.n_points*fraction)

        idx = np.random.choice(self.n_points, self.n_samples, replace=False).tolist()
        self.y_sampled = self.y[idx]
        self.X_sampled = self.X[idx]

        # Drop these from the original dataset
        self.training_data = self.data.iloc[idx]
        self.testing_data = self.data.drop(idx)
        
        # Define the objective function for the dual problem
        def objective(lambda_, X, y):
                      
            # Define the dual objective function to be minimized
            obj = - np.sum(lambda_) + 0.5 * np.sum(np.outer(lambda_ * y, lambda_ * y) * np.dot(X, X.T))
                
            return obj

        # Define the bounds for lambda_
        bounds = [(0, None)]

        # Define the constraints
        const = ({'type': 'eq', 'fun': lambda lambda_: np.dot(lambda_, self.y_sampled)},{'type': 'ineq', 'fun': lambda lambda_: lambda_})

        # Call the minimize function to solve the dual problem
        result = minimize(objective, np.zeros(self.n_samples), args=(self.X_sampled, self.y_sampled), bounds=bounds, constraints=const)

        # Extract the solution
        lambda_ = result.x

        # Compute the weight vector
        self.w = np.sum([lambda_[i] * self.X_sampled[i] * self.y_sampled[i] for i in range(self.n_samples)], axis=0)

        # Compute the bias term
        support_vectors = lambda_ > 1e-5
        self.b = np.mean(self.y_sampled[support_vectors] - np.dot(self.X_sampled[support_vectors], self.w))

        # Print the results
        print("Dual coefficients (lambda_):", lambda_)
        print("Weight vector (w):", self.w)
        print("Bias term (b):", self.b)

    # Predict the outcome of the remaining datapoints
    def predict(self):

        # Get the testing data in the correct format
        self.X_test = self.testing_data[["Feature 1","Feature 2"]].to_numpy()

        # Predict the outcome of value X using the hyperplane
        def hyperplane(X):
            y = np.dot(self.w, X) + self.b
            if y > 0:
                return 1
            else:
                return -1

        # Get values for a testing set and store them in a new dataframe
        self.predicted_data = self.testing_data.copy()
        self.predicted_data["label"] = [hyperplane(i) for i in self.X_test]

    # Visualize the hyperplane and datapoints
    def visualize(self):

        # Plot the hyperplane
        x_range = np.arange(self.X_min, self.X_max, 0.1)
        y = (-self.w[0] * x_range - self.b) / self.w[1]
        fig, ax = plt.subplots()
        ax.plot(x_range, y, '-0', label='hyperplane')
        
        # Create a scatter plot containing the training datapoints
        colors = {-1:'darkred', 1:'darkblue'}
        ax.scatter(self.training_data['Feature 1'], self.training_data['Feature 2'], c=self.training_data['label'].apply(lambda x: colors[x]), label="Training Data")

        # Create a scatter plot containing the predicted datapoints
        colors = {-1:'salmon', 1:'skyblue'}
        ax.scatter(self.predicted_data['Feature 1'], self.predicted_data['Feature 2'], c=self.predicted_data['label'].apply(lambda x: colors[x]), label="Predicted")

        # Create a scatter plot containing the wrongly predicted datapoints
        self.incorrect_data = pd.concat([self.predicted_data,self.testing_data]).drop_duplicates(keep=False)
        colors = {-1:'magenta', 1:'magenta'}
        ax.scatter(self.incorrect_data['Feature 1'], self.incorrect_data['Feature 2'], c=self.incorrect_data['label'].apply(lambda x: colors[x]), label="Incorrectly predicted")

        # Show plot
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.legend()
        plt.show()
