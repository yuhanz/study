# Throw a ball into a basket at a velocity
#  The ball is thrown from (0,0), and the basket is at (100, 10)
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

import matplotlib.pyplot as plt

start_location = np.array([0, 0])  # initial x,y
target_location = np.array([100, 10])  # target x,y
acceleration = np.array([0, -9.8])
steps = 10
time_per_step = 1  # how much time (seconds) each step represents
velocity = np.array([1, 1])  # my initial guess. minimize will adjust this value

def update(velocity, location):
  return location + velocity * time_per_step

def simulate(velocity, location, acceleration, steps):
  intermediat_locations = []
  for i in range(0,steps):
    location = update(velocity, location)
    intermediat_locations.append(location)
    velocity = velocity + acceleration * time_per_step;
  return location, intermediat_locations

def objective(params):
  velocity = params
  final_location, intermediat_locations = simulate(velocity, start_location, acceleration, steps)
  return np.mean((final_location - target_location)**2)

objective_and_grad = value_and_grad(objective)

result = minimize(objective_and_grad, velocity, jac=True, method='CG')
desired_velocity = result.x

print("Desired velocity is ", desired_velocity)

destination, intermediate_locations = simulate( np.array([10. , 45.1]), start_location, acceleration, 10)

print("result destination: ", destination)
print("intermediate_locations: ", intermediate_locations)

# plt.plot([start_location] + intermediate_locations)
x = list(map(lambda v: v[0], ([start_location] + intermediate_locations)))
y = list(map(lambda v: v[1], ([start_location] + intermediate_locations)))

plt.plot(x, y)
plt.show()
