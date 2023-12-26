import numpy as np
from scipy.optimize import minimize
import joblib
import matplotlib.pyplot as plt

# Load the saved model and scaler
model = joblib.load('energy_reg_model.joblib')
scaler = joblib.load('scaler.joblib')

# Load one row of data for initialization
initial_data = np.array([[30.00, 19.20, 7.03,  6.60,  7.00,   47.60,  19.79, 84.26,  48.90, 92.00]])

# Extract individual variables it might be more useful
[lights, T2, T6, T_out, Windspd, RH_1, T3, RH_6, RH_8, RH_out] = initial_data[:, :].flatten()
# print('lights: ', lights)
# print('T2: ', T2)
# print('T6: ', T6)
# print('T_out: ', T_out)
# print('Windspd: ', Windspd)
# print('RH_1: ', RH_1)
# print('T3: ', T3)
# print('RH_6: ', RH_6)
# print('RH_8: ', RH_8)
# print('RH_out: ', RH_out)

# Define controllable parameters and their suggested ranges
controllable_ranges = {

    'T2': (18, 22),
    'RH_1': (40, 60),
    'T3': (18, 22),
    'RH_8': (40, 60),
}

# Define MPC parameters
horizon = 3  # MPC prediction horizon
num_steps = 4  # Number of MPC steps

# Initialize arrays to store trajectories and objective values
state_trajectory = np.zeros((num_steps + 1, 4))
control_trajectory = np.zeros((num_steps, 4))
objective_values = []

# Define constraints
def constraint_fun(u):
    # Reshape the 1D array u to 2D (matrix)
    u_matrix = u.reshape((horizon, 4))
    
    # Extract individual control inputs for each time step
    T2_values = u_matrix[:, 0]
    RH_1_values = u_matrix[:, 1]
    T3_values = u_matrix[:, 2]
    RH_8_values = u_matrix[:, 3]

    # Define constraints based on controllable ranges
    T2_constraints = np.concatenate([T2_values - controllable_ranges['T2'][0], controllable_ranges['T2'][1] - T2_values])
    RH_1_constraints = np.concatenate([RH_1_values - controllable_ranges['RH_1'][0], controllable_ranges['RH_1'][1] - RH_1_values])
    T3_constraints = np.concatenate([T3_values - controllable_ranges['T3'][0], controllable_ranges['T3'][1] - T3_values])
    RH_8_constraints = np.concatenate([RH_8_values - controllable_ranges['RH_8'][0], controllable_ranges['RH_8'][1] - RH_8_values])

    # Combine all constraints into a single array
    all_constraints = np.concatenate([T2_constraints, RH_1_constraints, T3_constraints, RH_8_constraints])

    return all_constraints


# Helper function for objective
def objective_step(u, t):
    # Reshape the 1D array u to 2D (matrix)
    u_matrix = u.reshape((horizon, 4))
    
    full_state = [lights, u_matrix[t, 0], T6, T_out, Windspd, u_matrix[t, 1], u_matrix[t, 2], RH_6, u_matrix[t, 3], RH_out]
    scaled_full_state = scaler.transform([full_state])
    predicted_energy = model.predict(scaled_full_state)
    print('predicted_energy: ', predicted_energy)
    return predicted_energy[0]

# Define the objective function
def objective(u):
    full_states_errors = [(objective_step(u, t) - 55) ** 2 for t in range(horizon)]
    # print('full_states_errors: ', full_states_errors)
    return np.sum(full_states_errors)


# you can replace whatever update model that best describes your system dynamics
def update_state(current_state, optimal_input_step):
    updated_state = [
        current_state[0] + 0.00002*optimal_input_step[0],
        current_state[1] + 0.000002*optimal_input_step[1],
        current_state[2] + 0.00002*optimal_input_step[2],
        current_state[3] - 0.0002*optimal_input_step[3],
    ]
    return updated_state

current_state = [T2, RH_1, T3, RH_8]

# Example MPC loop
for step in range(num_steps):

    # Simulate current state using the provided initial state
    state_trajectory[step, :] = current_state

    # Define optimization variables
    # u = np.zeros((horizon, 4))  # Use a 1D array for optimization

    # IF YOU CHANGE HORIZON, YOU HAVE TO MAKE THESE COPIES LOOK COMPLETE TO HORIZON
    current_state = np.array([current_state,current_state,current_state])
    u_flat = current_state.flatten()

    # Solve the optimization problem using SciPy
    result = minimize(objective, u_flat, constraints={'type': 'ineq', 'fun': constraint_fun})
    # print('result: ', result)

    optimal_input = result.x.reshape((horizon, 4))  # Reshape the result to a matrix

    # Extract the optimal control input for the current step
    optimal_input_step = optimal_input[0, :]
    control_trajectory[step, :] = optimal_input_step

    # Update the state based on the optimal control input
    current_state = update_state(current_state[0], optimal_input_step)

    # Calculate and store the objective function value
    objective_values.append(result.fun)



# plotting and analysis...

# State and Control Trajectories Plot
plt.figure(figsize=(12, 8))
for i in range(4):
    plt.plot(state_trajectory[:, i], label=f'State {i}')
for i in range(4):
    plt.plot(control_trajectory[:, i], label=f'Control {i}', linestyle='--')
plt.xlabel('MPC Steps')
plt.ylabel('Values')
plt.title('State and Control Trajectories')
plt.legend()
plt.show()

# Objective Function Value Plot
plt.figure(figsize=(8, 6))
plt.plot(objective_values, marker='o')
plt.xlabel('MPC Steps')
plt.ylabel('Objective Function Value')
plt.title('Objective Function Evolution')
plt.show()

# State and Control Profiles at a Specific Time Step
time_step_to_plot = 3  # Choose a specific time step to plot

plt.figure(figsize=(12, 8))

# Plot the state profile at the chosen time step
plt.subplot(2, 1, 1)
plt.plot(state_trajectory[time_step_to_plot, :])
plt.title(f'State Profile at Time Step {time_step_to_plot}')

# Plot the control profile at the chosen time step
plt.subplot(2, 1, 2)
plt.plot(control_trajectory[time_step_to_plot, :], linestyle='--')
plt.title(f'Control Profile at Time Step {time_step_to_plot}')

plt.tight_layout()
plt.show()
