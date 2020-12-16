from math import *
import numpy as np
import time
from decimal import *

class MinAndMaxPositions():
	def __init__(self, minimum, maximum):
		self.minimum = minimum
		self.maximum = maximum

class Mesh:
	def __init__(self):
		self.vertices =[]
		self.faces = []

class MeshSubdivider:
	"""A class to read, write and split meshes"""

	def __init__(self, filename, output_dir, level, maximum_level, minAndMaxPositions):
		self.maximum_level = maximum_level
		self.level = level
		self.output_dir = output_dir

		self.mesh_s0 = Mesh()
		with open(filename, "r") as f:
			for line in f:

				s = line.split(" ")
				if s[0] == "v":
					pos = (float(s[1]), float(s[2]), float(s[3]))

					#if pos not in self.mesh_s0.vertices:
					#self.add_vertex(pos)
					self.mesh_s0.vertices.append(pos)
					#self.mesh_s0.vertices[pos] = len(self.mesh_s0.vertices)
						
				if s[0] == "f":
					vertex_ids = (int(s[1])-1, int(s[2])-1, int(s[3])-1) #subtract one for python indices
					self.mesh_s0.faces.append(vertex_ids)
					#self.add_face(vertex_ids)
					#self.mesh_s0.faces[vertex_ids] = len(self.mesh_s0.faces) #subtract one for python indices

		print("Read in file complete")
		self.minimum = minAndMaxPositions.minimum
		self.maximum = minAndMaxPositions.maximum

		self.grid_size = [(self.maximum[d]-self.minimum[d])/2**self.maximum_level for d in range(0,3)]
		print(self.grid_size)


		self.max_grid_index = 2**self.maximum_level-1
		self.octree_meshes = {}
		for x in range(0,2**self.level):
			for y in range(0,2**self.level):
				for z in range(0,2**self.level):
					grid_indices = f"{x}{y}{z}"
					self.octree_meshes[grid_indices] = Mesh()


	def grid_location(self, vertices, dimension):
		return [ floor((vertex[dimension] - self.minimum[dimension])/self.grid_size[dimension])*self.grid_size[dimension] + self.minimum[dimension] for vertex in vertices]

	def get_grid_index_for_vertex(self, vertex, dimension):
		#min necessary for when point is at max location
		temp = (vertex[dimension] - self.minimum[dimension])/self.grid_size[dimension]
		return min( floor((vertex[dimension] - self.minimum[dimension])/self.grid_size[dimension]), self.max_grid_index)

	def get_grid_indices_for_face_vertices(self, vertices):
		grid_indices = [-1,-1,-1]
		for dimension in range(0,3):
			not_on_border_ids = [index for index,on_border in enumerate(self.is_on_border(vertices,dimension)) if on_border==False]
			#if none are on a border or if three are on the same border, then proceed
			if len(not_on_border_ids)<3 and len(not_on_border_ids)>0:
				grid_indices[dimension] = self.get_grid_index_for_vertex(vertices[not_on_border_ids[0]], dimension)
			else:
				grid_indices[dimension] = self.get_grid_index_for_vertex(vertices[0], dimension)

		grid_indices = [floor(grid_indices[d] * (2**self.level / 2**self.maximum_level)) for d in range(0,3)] #since need every level to be able to be evenly split
		return f"{grid_indices[0]}{grid_indices[1]}{grid_indices[2]}"
	
	def is_on_border(self, vertices, dimension):
		on_border = []
		for vertex in vertices:
			if vertex[dimension] == self.minimum[dimension] or vertex[dimension] == self.maximum[dimension]:
				on_border.append(False)
			else: 
				on_border.append(((vertex[dimension] - self.minimum[dimension])/self.grid_size[dimension]) % 1 == 0)

		return on_border


	def get_average_vertex(self, vertices):
		avg = tuple([0.5*(vertices[0][d]+vertices[1][d]) for d in range(0,3)])  #[vertices[0][d]*0.5 + vertices[1][d]*0.5 for d in range(0,3)] #[vertices[0][d]+0.5*(vertices[1][d]-vertices[0][d]) for d in range(0,3)] #[0.5*(vertices[0][d] + vertices[1][d]) for d in range(0,3)];
		return avg
	
	def add_vertex(self, vertex):
		if vertex not in self.mesh_s0.vertices:
			self.mesh_s0.vertices.append(vertex)
		return self.mesh_s0.vertices.index(vertex)
		#self.mesh_s0.vertices.append(vertex)
		#return len(self.mesh_s0.vertices)-1

	def add_face(self, face):
		if face not in self.mesh_s0.faces:
			self.mesh_s0.faces.append(face)

	def split_face_in_two(self, face, vertices, dimension, is_on_border, grid_locations):
		#remove facef
		#print("split two")
		self.mesh_s0.faces.remove(face)

		#		0																					  0
		#      / \   one of these is on border, so will add an extra point (eg. if 0 is on border)   /|\
		#     1---2																					1-.-2
		vertices_to_process = [vertices[i] for i in range(0,3) if is_on_border[i]==False]
		intermediate_vertex =  self.get_intermediate_vertex(vertices_to_process, dimension, False, grid_locations)

		border_point = [i for i in range(0,3) if is_on_border[i]==True][0]

		#add new vertex
		intermediate_vertex_id = self.add_vertex(intermediate_vertex)

		#add new faces
		connects_to = [1,2,0]
		preceeded_by = [2, 0, 1]
		self.add_face( (face[ connects_to[border_point] ], intermediate_vertex_id, face[border_point]) )
		self.add_face( (face[border_point], intermediate_vertex_id, face[ preceeded_by[border_point] ]) )

	def split_face_in_four(self, face, vertices, dimension, grid_locations):
		self.mesh_s0.faces.remove(face)
		#print("split four")

		#     0      then becomes 4 triangles      0
		#    / \                                  /_\             
		#   /   \                                /\ /\
		#  1-----2                              1--v--2
		#
		if grid_locations[0] == grid_locations[1]:
			vertex_in_own_grid = 2
		elif grid_locations[1] == grid_locations[2]:
			vertex_in_own_grid = 0
		else:
			vertex_in_own_grid = 1

		intermediate_vertices = []
		#0: 0->1, 1: 1->2, 2: 2->0
		ids_to_split = {0 : [0,1], 1: [1,2], 2: [2,0]}
		intermediate_vertex_ids = []
		for i in range(0,3):
			current_ids_to_split = ids_to_split[i]
			take_average = vertex_in_own_grid not in current_ids_to_split # then the two cross a grid boundary
			vertices_to_process = [vertices[j] for j in current_ids_to_split]
			intermediate_vertex = self.get_intermediate_vertex(vertices_to_process, dimension, take_average,grid_locations)
			intermediate_vertex_id = self.add_vertex(intermediate_vertex)
			intermediate_vertex_ids.append(intermediate_vertex_id)

			if take_average:
				#need to split edge that this touches so that all vertices are part of at least 3 triangles
				vertex_ids = [face[current_ids_to_split[0]], face[current_ids_to_split[1] ] ]
				matched = False
				for f in self.mesh_s0.faces:
					if vertex_ids[0] in f and vertex_ids[1] in f: #then this face needs to be split
						self.mesh_s0.faces.remove(f)
						point_opposite_edge = [index for index,p in enumerate(f) if p not in [vertex_ids[0], vertex_ids[1]]][0]
						connects_to = [1,2,0]
						preceeded_by = [2, 0, 1]
						self.add_face( ( f[connects_to[point_opposite_edge] ], intermediate_vertex_id, f[point_opposite_edge] ) )
						self.add_face( ( f[point_opposite_edge], intermediate_vertex_id, f[ preceeded_by[point_opposite_edge]] ) )
						matched = True
						break
				if matched==False:
					print("failed")


			#print(f"{take_average} {intermediate_vertex}")
		
		#self.add_face([face[2], face[0], face[1]])
		self.add_face( (face[0], intermediate_vertex_ids[0], intermediate_vertex_ids[2]  ) )
		self.add_face( (face[1], intermediate_vertex_ids[1], intermediate_vertex_ids[0]  ) )
		self.add_face( (face[2], intermediate_vertex_ids[2], intermediate_vertex_ids[1] ) )
		self.add_face( (intermediate_vertex_ids[0], intermediate_vertex_ids[1], intermediate_vertex_ids[2] ) )

		# old_triangle_ids = [[face[0], face[1],face[2] ]]
		# for ids in old_triangle_ids:
		# 	for id in ids:
		# 		print(f"v {self.mesh_s0.vertices[id][0]} {self.mesh_s0.vertices[id][1]} {self.mesh_s0.vertices[id][2]}")

		# for f in [ [1, 2,3]]:
		# 	print(f"f {f[0]} {f[1]} {f[2]}")

		# print("\n")
		# new_triangle_ids = [[face[0], intermediate_vertex_ids[0], intermediate_vertex_ids[2] ], [face[1], intermediate_vertex_ids[1], intermediate_vertex_ids[0]  ], [face[2], intermediate_vertex_ids[2], intermediate_vertex_ids[1] ], [intermediate_vertex_ids[0], intermediate_vertex_ids[1], intermediate_vertex_ids[2] ] ]
		# for ids in new_triangle_ids:
		# 	for id in ids:
		# 		print(f"v {self.mesh_s0.vertices[id][0]} {self.mesh_s0.vertices[id][1]} {self.mesh_s0.vertices[id][2]}")

		# for f in [ [1, 2,3],[4,5,6],[7,8,9],[10,11,12] ]:
		# 	print(f"f {f[0]} {f[1]} {f[2]}")



	def get_intermediate_vertex(self, vertices, dimension, take_average, grid_locations):
		if take_average: #take average
			intermediate_vertex = self.get_average_vertex(vertices)
		else: #intersection with grid
			point_on_plane = [0,0,0]
			plane_normal = [0,0,0]
			point_on_plane[dimension] = max(grid_locations)
			plane_normal[dimension] = 1

			intermediate_vertex = self.my_line_plane_intersection(
				point_on_plane,
				plane_normal,
				dimension,
				vertices[0],
				vertices[1])

		return intermediate_vertex


	def my_line_plane_intersection(self, point_on_plane, plane_normal, normal_direction, line_point_one, line_point_two):
		#https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
		#start = time.time()
		point_on_line = line_point_one
		line_direction = [ line_point_one[dimension] - line_point_two[dimension] for dimension in range(0,3)]

		d = (point_on_plane[normal_direction]-point_on_line[normal_direction])/line_direction[normal_direction]
		p = tuple([ point_on_line[dimension] + line_direction[dimension]*d for dimension in range(0,3)])
		#print(time.time()-start)
		return p


	def prepare_mesh_for_single_axis_subdivision(self, dimension): #0->x, 1->y, 2->z
		for face in reversed(self.mesh_s0.faces):
			vertices = [self.mesh_s0.vertices[face[0]], self.mesh_s0.vertices[face[1]], self.mesh_s0.vertices[face[2]] ]
			grid_locations = self.grid_location(vertices, dimension) #, self.grid_location(v1, dimension, grid_size), self.grid_location(v2, dimension, grid_size)]
			if not all(ele == grid_locations[0] for ele in grid_locations):

				is_on_border = self.is_on_border(vertices, dimension)
				non_border_vertex_ids = [i for i in range(0,3) if is_on_border[i]==False]
				#print(grid_locations)
				#print(is_on_border) 
				if sum(is_on_border) == 1 and grid_locations[non_border_vertex_ids[0]] != grid_locations[non_border_vertex_ids[1]]: #then triangle straddles border
					self.split_face_in_two(face, vertices, dimension, is_on_border, grid_locations)

				if sum(is_on_border) == 0:
					self.split_face_in_four(face,vertices, dimension, grid_locations)
					
	def prepare_mesh_for_octree_subdivision(self):
		for dimension in range(0,3):
			self.prepare_mesh_for_single_axis_subdivision(dimension)
			print(f"Subdivision {dimension} complete")

	def split_mesh(self):
		for face in self.mesh_s0.faces:
			face_in_mesh = []
			vertices = (self.mesh_s0.vertices[face[0]], self.mesh_s0.vertices[face[1]], self.mesh_s0.vertices[face[2]] )
			grid_indices = self.get_grid_indices_for_face_vertices(vertices)
			for vertex in vertices:
				#commented out some stuff here cuz duplicating faces is much faster than checking if vertices are present, and use another code to compress anyway
				if vertex not in self.octree_meshes[grid_indices].vertices:
					self.octree_meshes[grid_indices].vertices.append(vertex)
				vertex_id = self.octree_meshes[grid_indices].vertices.index(vertex)
				face_in_mesh.append(vertex_id)
			self.octree_meshes[grid_indices].faces.append(face_in_mesh)

		print("Split mesh complete")

	def quantize_vertex(self, vertex, position):
		#assume 16 bit
		#POINTLESS?
		quantization = (2**16)-1
		chunk_shape = [14031.0, 1584.0, 5555.0]
		current_chunk_shape = [chunk_shape[d]/(2**self.level) for d in range(0,3)]

		vertex =  [round(((vertex[d]-self.minimum[d])/current_chunk_shape[d] - position[d])*quantization) for d in range(0,3)]
		return vertex

	def write_meshes(self):
		#with open(f"test/mySubdivide/s0/original.obj", 'w') as outfile:
		#	for vertex in self.mesh_s0.vertices:
		#		outfile.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
		#	for face in self.mesh_s0.faces:
		#		outfile.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n") #add one for obj file
		for key in self.octree_meshes.keys():
			with open(f"{self.output_dir}/{key}.obj", 'w') as outfile:
				for vertex in self.octree_meshes[key].vertices:
					position = [int(d) for d in key]
					vertex = [vertex[d]-self.minimum[d] for d in range(0,3)] #self.quantize_vertex(vertex, position) #[vertex[d]/current_chunk_shape[d] for d in range(0,3)]
					#print(f"{key} {vertex}")
					outfile.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
				for face in self.octree_meshes[key].faces:
					outfile.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n") #add one for obj file

		print("Write meshes complete")

	def octree_subdivision(self):
		self.prepare_mesh_for_octree_subdivision()		
		self.split_mesh()
		self.write_meshes()
#	def octree_subidvision:
#		for x in

def get_min_and_max(directory_prefix, scales, mesh):
	minimum = [10E9,10E9,10E9]
	maximum = [0,0,0]
	for scale in scales:
		filename = f"{directory_prefix}{scale}/{mesh}"
		with open(filename, "r") as f:
			for line in f:
				s = line.split(" ")
				if s[0] == "v":
					pos = [float(s[1]), float(s[2]), float(s[3])]
					for i in range(0,3):
						#floor minimum and ceil maximum to nearest whole number to prevent floating point things later
						minimum[i] = min(minimum[i], floor(pos[i])-1) # one for padding
						maximum[i] = max(maximum[i], ceil(pos[i])+1)
	print(minimum)
	print(maximum)
	return MinAndMaxPositions(minimum, maximum)

#minAndMaxPositions = get_min_and_max("cube/s", [0], "cube.obj");

#mesh_subdivider = MeshSubdivider("cube/s0/cube.obj","cube/divided/",0, 1, minAndMaxPositions) 
#mesh_subdivider.octree_subdivision()

minAndMaxPositions = get_min_and_max("test/mito_obj_meshes_s", [0,1,2], "345809856042.obj");

mesh_subdivider = MeshSubdivider("test/mito_obj_meshes_s2/345809856042.obj","test/mySubdivide/s2",0,2, minAndMaxPositions) 
mesh_subdivider.octree_subdivision()

#mesh_subdivider = MeshSubdivider("test/mito_obj_meshes_s1/345809856042.obj","test/mySubdivide/s1",1,2, minAndMaxPositions) 
#mesh_subdivider.octree_subdivision()

#mesh_subdivider = MeshSubdivider("test/mito_obj_meshes_s0/345809856042.obj","test/mySubdivide/s0",2,2, minAndMaxPositions)
#mesh_subdivider.octree_subdivision()

print(mesh_subdivider.minimum)
print(mesh_subdivider.maximum)
