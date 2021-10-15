# multiresolutionMeshes
git clone --recursive https://github.com/davidackerman/multiresolution-mesh-creator.git
conda env update -n my_env --file ENV.yaml
conda env update -n multiresolution_mesh_creator --file multiresolution_mesh_creator.yml
conda activate multiresolution_mesh_creator
cd dvidutils follow instructions
cd pyfqmr-Fast-Quadric-Mesh-Reduction
python setup.py install
python create_multiresolution_meshes.py -i test_meshes/ -o test_meshes_output/ -n 5 -b 4

Given a set of multiresolution meshes for an object in obj format, generate a multiresolution mesh in the [neuroglancer precomputed format](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md), based on [this comment](https://github.com/google/neuroglancer/issues/272#issuecomment-752212014). Uses a custom draco exporter based on [this suggestion]( https://github.com/google/neuroglancer/issues/266#issuecomment-739601142).

In the example in the code, input meshes for object id 345809856042 are in directories structured as `test/mito_obj_meshes_s{lod}/345809856042.obj`, where lod is the level of detail.

Running `multiresolution_sharding.py` will read in the meshes and output the precomputed mesh and corresponding json.

NOTE: This is a work in progress, seams may be exist in meshes after subdivision.

![Demo](recording/recording.gif)


