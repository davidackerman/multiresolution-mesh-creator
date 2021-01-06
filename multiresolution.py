import struct
import json
import os

class Fragment():
	def __init__(self, position, size_in_bytes):
		self.position = position
		self.size_in_bytes = size_in_bytes

def cmp_zorder(lhs, rhs) -> bool:
    """Compare z-ordering."""
    # Assume lhs and rhs array-like objects of indices.
    assert len(lhs) == len(rhs)
    # Will contain the most significant dimension.
    msd = 2
    # Loop over the other dimensions.
    for dim in [1, 0]:#len(lhs)):
        # Check if the current dimension is more significant
        # by comparing the most significant bits.
        if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
            msd = dim
    return lhs[msd] < rhs[msd]

def less_msb(x: int, y: int) -> bool:
	return x < y and x < (x ^ y)

def get_z_ordered_list(unordered_list):
	ordered_list = [unordered_list[0]]
	for i in range(1,len(unordered_list)):
		inserted = False
		for j in range(0,len(ordered_list)):
			element_to_insert = unordered_list[i]
			should_insert = cmp_zorder(element_to_insert, ordered_list[j])
			if should_insert:
				inserted = True
				ordered_list.insert(j, element_to_insert)
				break
		if not inserted:
			ordered_list.append(element_to_insert)
	return ordered_list

def get_fragment_order(divisions):
	unordered_list = []

	for x in range(0, divisions):
		for y in range(0, divisions):
			for z in range(0, divisions):
				unordered_list.append([x,y,z])

	return get_z_ordered_list(unordered_list)

#info file
info_dict = {"@type": "neuroglancer_multilod_draco",
"vertex_quantization_bits": 10,
"transform": [1,0,0,0,0,1,0,0,0,0,1,0],
"lod_scale_multiplier": 1,
}
with open('test/multiresolution/info', 'w') as json_file:
	json.dump(info_dict, json_file)

#for filename in *.obj; do filename="${filename%.*}"; draco_encoder -i $filename.obj -o $filename.drc -qp 16; done

# lod 0: highest res, smallest chunks, most chunks. chunk shape of chunk_shape
# lod 1: 
num_divisions_per_dimension_lod0 = 4
minimum = [19388, 4147, 18209]
maximum = [33404, 5729, 23743]


lods = [0, 1,2]
chunk_shape = [3508.25, 396.75, 1389.25]#[(maximum[d]-minimum[d])/num_divisions_per_dimension_lod0 for d in range(0,3)]
chunk_shape = [chunk_shape[d]*(2**min(lods)) for d in range(0,3) ]
fragments_per_lod = []
cat_command = "cat"
for lod in lods:
	ordered_fragments = get_fragment_order(int(num_divisions_per_dimension_lod0/(2**lod)))
	print(ordered_fragments)
	scale = lod
	fragments_that_exist = []
	fragment_directory = f"test/mySubdivide/s{scale}/"
	os.system(f"rm {fragment_directory}/*.drc")
	for fragment in ordered_fragments:
			print(fragment)
			current_chunk_shape = [chunk_shape[d]*2**lod for d in range(0,3)]
		#if fragment == [0,0,0] :#or fragment == [1,0,0]:
			os.system(f"./dracoC/convertToQuantizedDracoMesh {fragment_directory} {current_chunk_shape[0]} {current_chunk_shape[1]} {current_chunk_shape[2]} {fragment[0]} {fragment[1]} {fragment[2]}")
			fragment_path = f"{fragment_directory}/{fragment[0]}{fragment[1]}{fragment[2]}.drc"
			if os.path.isfile(f"{fragment_path}"): #else is empty fragment
				#fragments_that_exist.append(Fragment([count,0,0], os.path.getsize(fragment_path)))
				fragments_that_exist.append(Fragment(fragment, os.path.getsize(f"{fragment_path}")))
				cat_command=cat_command+f" {fragment_path}"
			#else:
			#	fragments_that_exist.append(Fragment(fragment, 0))
	fragments_per_lod.append(fragments_that_exist)

cat_command=cat_command+" > test/multiresolution/345809856042"
print(cat_command)
os.system(cat_command)
#manifest file

print(chunk_shape)
with open(f"test/multiresolution/345809856042.index",'wb') as outfile:
	#want highest resolution first
	num_fragments_per_lod = [len(fragments) for fragments in fragments_per_lod ]
	print(num_fragments_per_lod)
	buf =  struct.pack('<3f',*chunk_shape) #chunk shape
	buf += struct.pack('<3f',0,0,0) #grid orgin
	buf += struct.pack('<1I',len(lods)) #number of levels of detail (lods)
	buf += struct.pack(f"<{len(lods)}f",*[2**lod for lod in lods])#*[2**(len(lods)-lod+1) for lod in lods]) #lod scales
	buf += struct.pack(f"<{3*len(lods)}f",*([0,0,0]*len(lods))) #vertex offsets
	buf += struct.pack(f"<{len(lods)}I",*num_fragments_per_lod) #num fragments per lod

	#for each lod
	for fragments_in_current_lod in fragments_per_lod:
		for d in range(0,3):
			for fragment in fragments_in_current_lod:
				buf += struct.pack('<I',fragment.position[d])
		for fragment in fragments_in_current_lod:
			buf += struct.pack('<I',fragment.size_in_bytes)
	outfile.write(buf)


# with open(f"test/multiresolution/345809856042.index",'wb') as outfile:
# 	buf =  struct.pack('<3f',5000,5000,5000) #chunk shape
# 	buf += struct.pack('<3f',0,0,0) #grid orgin
# 	buf += struct.pack('<1I',3) #number of levels of detail (lods)
# 	buf += struct.pack('<3f',1,2,4) #lod scales
# 	buf += struct.pack('<9f',0,0,0,0,0,0,0,0,0) #vertex offsets
# 	buf += struct.pack('<3I',1,1,1) #num fragments per lod
# 	#for each lod
# 	buf += struct.pack('<3I',0,0,0) #fragment positions
# 	buf += struct.pack('<1I',1470257)
# 	buf += struct.pack('<3I',0,0,0) #fragment positions
# 	buf += struct.pack('<1I',422276)
# 	buf += struct.pack('<3I',0,0,0) #fragment positions
# 	buf += struct.pack('<1I',121239)
# 	#500384) #fragment offsets
# 	outfile.write(buf)