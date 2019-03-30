from PIL import Image
import retro
import numpy as np
from image_util import detection

env = retro.make(game='MegaMan2-Nes', record='.')


previous_screen = env.reset()

# either left or right
actionRight = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])  # forward
actionLeft = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])  # backward

# either jump or release
actionJump = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]) # jump

# either fire or release
actionFire = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])  # fire

actionRelease = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]) # release
#action = env.action_space.sample()

frames = 30

# basic inputs
# [brake, steer, press_fuel]
# brake: if brake > 0: stop moving left or right
#       if brake <= 0: let move decide direction
# steer : if steer > 0: move left
#        if steer <= 0: move right
# press_fuel: adding this amount to jump_fuel. This amount can be negative

jump_fuel = 0   # when jump fuel is none zero. keep holding the jump

action = actionRelease

for i in range(1,frames):

    # print(action)
    _obs, _rew, done, _info = env.step(action)

    if done:
        break
    if i % 4 == 0:
        shiftValue = detection.detect_horizontal_shift_all_rows(previous_screen, _obs)
        print('shift: ', shiftValue)

        # TODO: decide [brake, steer, press_fuel]
        [brake, steer, press_fuel] = [0, -1, 0]
        jump_fuel += press_fuel

        jump = actionJump * [1 if jump_fuel > 0 else 0]
        fire = actionFire * (i%2)
        left = actionLeft * (steer > 0)
        right = actionRight * (steer <= 0)
        action = fire + left + right + jump

        jump_fuel = 0 if jump_fuel <= 0 else jump_fuel - 1

        previous_screen = _obs
        img = Image.fromarray(_obs, 'RGB')
        img.show()
