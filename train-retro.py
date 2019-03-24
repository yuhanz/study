from PIL import Image
import retro
import numpy as np
from image_util import detection

env = retro.make(game='MegaMan2-Nes', record='.')


previous_screen = env.reset()

actionFire = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])  # fire
actionForward = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])  # forward
#action1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]) # jump
actionRelease = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]) # release
#action = env.action_space.sample()

frames = 20

for i in range(1,frames):
    action = np.logical_or(actionForward, [actionFire, actionRelease][i%2])
    # print(action)
    _obs, _rew, done, _info = env.step(action)

    if done:
        break
    if i % 4 == 0:
        shiftValue = detection.detect_horizontal_shift_all_rows(previous_screen, _obs)
        print('shift: ', shiftValue)
        previous_screen = _obs
        img = Image.fromarray(_obs, 'RGB')
        img.show()
