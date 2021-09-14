# Throw a ball into a basket at a velocity
#  The ball is thrown from (0,0), and the basket is at (100, 10)
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from imageio import imread


flag = imread('/Users/yzhang/Olympic_flag.png')
target_location = (abs((flag - 255)).sum(axis=2) > 0) *255

start_location = np.zeros([160, 240])
start_location[:20, :] = 255       # start with content at the top


steps = 200
time_per_step = 1  # how much time (seconds) each step represents
# initial guess. minimize will adjust this value
velocity_x = np.ones([160, 240])
velocity_y = np.ones([160, 240])

def update(velocity_x, velocity_y, location):
  vs = velocity_x + velocity_y
  normalized_vx = velocity_x / vs
  normalized_vy = velocity_y / vs
  move_x = normalized_vx * location
  move_y = normalized_vy * location
  move_x_positive = np.roll(move_x * (move_x > 0) + np.zeros(move_x.shape), 1, axis=1)
  move_x_negative = np.roll(move_x * (move_x < 0) + np.zeros(move_x.shape), -1, axis=1)
  move_y_positive = np.roll(move_y * (move_y > 0) + np.zeros(move_y.shape), 1, axis=0)
  move_y_negative = np.roll(move_y * (move_y < 0) + np.zeros(move_y.shape), -1, axis=0)
  new_location = np.zeros(location.shape)
  new_location = move_x_positive + move_x_negative + move_y_positive + move_y_negative
  return new_location

def simulate(velocity_x, velocity_y, location, steps, collectIntermediate = False):
  intermediate_locations = []
  for i in range(0,steps):
    location = update(velocity_x, velocity_y, location)
    if(collectIntermediate):
        intermediate_locations.append(location.copy())
  return location, intermediate_locations

def objective(params):
  velocity_x = np.reshape(params[:160*240], [160,240])
  velocity_y = np.reshape(params[160*240:], [160,240])
  final_location, intermediate_locations = simulate(velocity_x, velocity_y, start_location, steps)
  return np.mean((final_location - target_location)**2)

objective_and_grad = value_and_grad(objective)

result = minimize(objective_and_grad, np.concatenate((velocity_x, velocity_y)), jac=True, method='CG')

desired_velocity_x = result.x[:160 * 240].reshape(160, 240)
desired_velocity_y = result.x[160 * 240:].reshape(160, 240)

# to run the objective function with trained parameters
objective_and_grad(np.concatenate((desired_velocity_x.reshape(160*240), desired_velocity_y.reshape(160*240))))

destination, intermediate_locations = simulate( desired_velocity_x, desired_velocity_y, start_location, steps, collectIntermediate=True)

print("result destination: ", destination)
print("intermediate_locations: ", intermediate_locations)

plt.imshow(destination)
plt.show()

plt.imshow(intermediate_locations[0])
plt.show()
plt.imshow(intermediate_locations[10])
plt.show()
plt.imshow(intermediate_locations[15])
plt.show()
plt.imshow(intermediate_locations[19])
plt.show()
