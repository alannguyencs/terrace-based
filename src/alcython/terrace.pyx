import numpy as np
cimport cython
cimport libcpp
cimport libcpp.queue
cimport libcpp.set
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc
from collections import defaultdict, OrderedDict
import time

cdef extern from "<algorithm>" namespace "std":
	void std_sort "std::sort" [iter](iter first, iter last)

DTYPE = np.intc

HEIGHT = 256
WIDTH = 256
cdef (pair [int, int])[65536] NUM2PAIRS
cdef int[256][256] PAIRS2NUM
cdef int [8] idy = [0, 1, 1, 1, 0, -1, -1, -1]
cdef int [8] idx = [1, 1, 0, -1, -1, -1, 0, 1]
cdef int [9][8] directions = [[1, 2, 3, 4, 5, 6, 7, 8],
							[4, 5, 6, 0, 0, 0, 0, 0],
							[4, 5, 6, 7, 8, 0, 0, 0],
							[6, 7, 8, 0, 0, 0, 0, 0],
							[1, 2, 6, 7, 8, 0, 0, 0],
							[1, 2, 8, 0, 0, 0, 0, 0],
							[1, 2, 3, 4, 8, 0, 0, 0],
							[2, 3, 4, 0, 0, 0, 0, 0],
							[2, 3, 4, 5, 6, 0, 0, 0]]

cdef int [9] arr_size = [8, 3, 5, 3, 5, 3, 5, 3, 5]
cdef int [9] opposite = [0, 5, 6, 7, 8, 1, 2, 3, 4]


#init pairs
for i in range(HEIGHT):
	for j in range(WIDTH):
		PAIRS2NUM[i][j] = i * HEIGHT + j
		NUM2PAIRS[PAIRS2NUM[i][j]] = pair[int, int](i, j)

@cython.boundscheck(False)
@cython.wraparound(False)


def produce_instance_map(int[:, ::1] terrace_map, int top_level, int min_instance_size):
	#return pixel_id
	cdef int[256][256] pixel_id
	cdef int BG_LEVEL = 0
	cdef int UG_LEVEL = -1
	cdef int MAX_ITENSITY = 999
	cdef float min_distance = 0
	cdef int min_idx = 0
	cdef float distance = 0

	cdef int[256][256] from_id  #from which direction 1-8
	cdef (vector[int])[20] seeds #maximum 20 instances
	cdef vector[int] v
	cdef vector[int] level_seeds

	cdef int run_id = 1
	cdef libcpp.queue.queue[int] q
	cdef full_expand = 0

	def expand_seed(int bi, int bj, int from_id_):
		cdef int cnt_ = arr_size[from_id_]
		for dir_id in range(arr_size[from_id_]):
			pi = bi + idy[directions[from_id_][dir_id] - 1]
			pj = bj + idx[directions[from_id_][dir_id] - 1]

			if terrace_map[pi, pj] >= top_level:				
				if pixel_id[pi][pj] == BG_LEVEL:
					q.push(PAIRS2NUM[pi][pj])
					pixel_id[pi][pj] = UG_LEVEL
					from_id[pi][pj] = opposite[directions[from_id_][dir_id]]
					cnt_ -= 1
		return cnt_

	def expand_outer(int bi, int bj, int from_id_, int current_level, int instance_id):
		cdef int cnt_ = arr_size[from_id_]
		for dir_id in range(arr_size[from_id_]):
			pi = bi + idy[directions[from_id_][dir_id] - 1]
			pj = bj + idx[directions[from_id_][dir_id] - 1]

			if terrace_map[pi, pj] >= current_level:				
				if pixel_id[pi][pj] <= BG_LEVEL:
					q.push(PAIRS2NUM[pi][pj])
					pixel_id[pi][pj] = instance_id
					from_id[pi][pj] = opposite[directions[from_id_][dir_id]]

					cnt_ -= 1
		return cnt_

	def propagate_at_level(vector[int] current_seeds, int current_level):
		cdef vector[int] next_seeds
		for vi in range(current_seeds.size()):
			q.push(current_seeds[vi])

		while not q.empty():
			p = q.front()
			(bi, bj) = NUM2PAIRS[p]
			q.pop()

			full_expand = expand_outer(bi, bj, from_id[bi][bj], current_level, pixel_id[bi][bj])
			if full_expand > 0: next_seeds.push_back(PAIRS2NUM[bi][bj])
		# print (current_level, next_seeds.size())
		return next_seeds


	# t0 = time.time()
	#step 0: init instance label = BG_LEVEL
	# for i in range(HEIGHT):
	# 	for j in range(WIDTH):
	# 		pixel_id[i][j] = BG_LEVEL
			

	# t1 = time.time()
	#step: collect seeds
	for i in range(HEIGHT):
		for j in range(WIDTH):
			if terrace_map[i, j] >= top_level and pixel_id[i][j] == BG_LEVEL:
				# print ("run", run_id, i, j)
				q.push(PAIRS2NUM[i][j])
				pixel_id[i][j] = UG_LEVEL
				v.clear()
				while not q.empty():
					p = q.front()
					(bi, bj) = NUM2PAIRS[p]
					# print ("inqueue", bi, bj)
					v.push_back(PAIRS2NUM[bi][bj])
					q.pop()

					full_expand = expand_seed(bi, bj, from_id[bi][bj])
					if full_expand > 0: seeds[run_id].push_back(PAIRS2NUM[bi][bj])

				#assign instance labels
				# print (v.size())
				if v.size() >= min_instance_size:
					for vi in range(v.size()):
						(bi, bj) = NUM2PAIRS[v[vi]]
						pixel_id[bi][bj] = run_id
						# centroid_path[bi][bj][run_id] = 0
					# centroid_mass[run_id] = v.size()
					run_id += 1
				else:
					seeds[run_id].clear()


	# t2 = time.time()
	# print ("seeding time: ", t2 - t1)
	#step: compute distance to centroids
	for id in range(1, run_id):
		for vi in range(seeds[id].size()):
			level_seeds.push_back(seeds[id][vi])

	# print ("start propagating", level_seeds.size())
	for contour_level in range(top_level-1, BG_LEVEL, -1):
		level_seeds = propagate_at_level(level_seeds, contour_level)


	# t3 = time.time()
	# [t0, t1, t2, t3] = [int(round(t * 1000, 0)) for t in [t0, t1, t2, t3]]
	# print ("{}:{}:{}".format(t1 - t0, t2 - t1, t3 - t2))

	return np.asarray(pixel_id)

		




