# Throw a ball into a basket at a velocity
#  The ball is thrown from (0,0), and the basket is at (100, 10)
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from imageio import imread


flag = imread('/Users/yzhang/Olympic_flag.png')
target_location = (abs((flag - 255)).sum(axis=2) > 0) *255

start_location = np.zeros([1280, 1920])
start_location[0:640, :] = 255       # start with content at the top


steps = 20
time_per_step = 1  # how much time (seconds) each step represents
# initial guess. minimize will adjust this value
velocity_x = np.ones([1280, 1920])
velocity_y = np.ones([1280, 1920])


def update(velocity_x, velocity_y, location):
  new_location = location - velocity_x
  new_location = new_location - velocity_y
  new_location = new_location - np.sqrt(velocity_x **2 + velocity_y ** 2) / 2
  update_x_positive = np.roll(velocity_x * (velocity_x > 0) + np.zeros(velocity_x.shape), 1)
  update_x_negative = np.roll(velocity_x * (velocity_x < 0) + np.zeros(velocity_x.shape), -1)
  update_y_positive = np.roll(velocity_y * (velocity_y > 0) + np.zeros(velocity_y.shape), 1, axis=0)
  update_y_negative = np.roll(velocity_y * (velocity_y < 0) + np.zeros(velocity_y.shape), -1, axis=0)
  new_location = new_location + update_x_positive
  new_location = new_location + update_x_negative
  new_location = new_location + update_y_positive
  new_location = new_location + update_y_negative
  return new_location

def simulate(velocity_x, velocity_y, location, steps):
  intermediat_locations = []
  for i in range(0,steps):
    location = update(velocity_x, velocity_y, location)
    intermediat_locations.append(location)
  return location, intermediat_locations

def objective(params):
  velocity_x = np.reshape(params[:1280*1920], [1280,1920])
  velocity_y = np.reshape(params[1280*1920:], [1280,1920])
  final_location, intermediat_locations = simulate(velocity_x, velocity_y, start_location, steps)
  return np.mean((final_location - target_location)**2)

objective_and_grad = value_and_grad(objective)

result = minimize(objective_and_grad, np.concatenate((velocity_x, velocity_y)), jac=True, method='CG')

desired_velocity_x = result.x[:1280 * 1920].reshape(1280, 1920)
desired_velocity_y = result.x[1280 * 1920:].reshape(1280, 1920)

destination, intermediate_locations = simulate( desired_velocity_x, desired_velocity_y, start_location, steps)

print("result destination: ", destination)
print("intermediate_locations: ", intermediate_locations)

# plt.plot([start_location] + intermediate_locations)
x = list(map(lambda v: v[0], ([start_location] + intermediate_locations)))
y = list(map(lambda v: v[1], ([start_location] + intermediate_locations)))

plt.plot(x, y)
plt.show()
