from constants import *
import queue
import numpy as np
from alcython import terrace
from collections import defaultdict, OrderedDict
from utils import util_color

BG_LEVEL = 0
FOOD_LEVEL = 1
SEED_LEVEL = 2
PADDING = 2


max_num_instances = 16
colors = np.array([list(util_color.hsv2rgb(_id / max_num_instances, 1)) for _id in range(max_num_instances)])


def save_instance_map(terrace_map, vis_path):
	colors = np.array(util_color.BBOX_COLORS)
	terrace_map_img = terrace_map.copy()
	terrace_map = terrace_map.astype(np.intc)

	terrace_map[:, 0] = 0
	terrace_map[:, -1] = 0
	terrace_map[0, :] = 0
	terrace_map[-1, :] = 0

	tmp = time.time()
	pixel_id = terrace.produce_instance_map(terrace_map=terrace_map, top_level=4, min_instance_size=32)
	finish_time = time.time()
	

	if np.max(terrace_map_img) > 0:
		terrace_map_img *= int(255 / np.max(terrace_map_img))
		terrace_map_img = Image.fromarray(terrace_map_img.astype('uint8'), 'L')
		terrace_map_img.save(vis_path.replace('.png', '_terrace.png'))


	num_instances = int(np.max(pixel_id))
	if num_instances >= 16:
		colors = np.array([list(util_color.hsv2rgb(_id / (num_instances+1), 1)) for _id in range((num_instances+1))])

	instance_map = np.zeros((pixel_id.shape[0], pixel_id.shape[1], 3))
	for instance_id in range(1, num_instances + 1):
		instance_map[pixel_id==instance_id] = colors[instance_id]

	instance_map = Image.fromarray(instance_map.astype('uint8'), 'RGB')
	instance_map.save(vis_path)

	return finish_time - tmp


