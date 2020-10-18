from PIL import Image
import sys
import math

FILE_NAME = sys.argv[1]
OUTPUT_PREFIX = sys.argv[2]

image = Image.open(FILE_NAME)
w, h = image.size
ratio = w / float(h)

TARGET_W = 160
TARGET_H = 160
TARGET_RATIO = TARGET_W / float(TARGET_H);

images = []

if ratio > TARGET_RATIO:
    width = int(TARGET_H * ratio)
    print("ratio", ratio)
    print("#", math.ceil(width / float(TARGET_W)))
    print("width, W:", width, TARGET_W)
    image = image.resize((width, TARGET_H))
    for i in range(0, math.ceil(width / float(TARGET_W))):
        startWidth = i * TARGET_W
        if(startWidth + TARGET_W > image.size[0]):
            startWidth = image.size[0] - TARGET_W
        # import pdb; pdb.set_trace()
        images = images + [image.crop((startWidth, 0, startWidth + TARGET_W, TARGET_H))]
else:
    height = int(TARGET_W / ratio)
    print("ratio", ratio)
    print("#", math.ceil(height / float(TARGET_H)))
    print("height, H:", height, TARGET_H)
    # import pdb; pdb.set_trace()
    image = image.resize((TARGET_W, height))
    for i in range(0, math.ceil(height / float(TARGET_H))):
        startHeight = i * TARGET_H
        if(startHeight + TARGET_H > image.size[1]):
            startHeight = image.size[1] - TARGET_H
        images = images + [image.crop((0, startHeight, TARGET_W, startHeight + TARGET_H))]

for index, image in enumerate(images):
    print('writing ', index)
    # import pdb; pdb.set_trace()
    image.save('{}image{}.png'.format(OUTPUT_PREFIX, index), "PNG")
