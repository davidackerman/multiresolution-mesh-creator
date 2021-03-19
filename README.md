# multiresolutionMeshes

Given a set of multiresolution meshes for an object in obj format, generate a multiresolution mesh in the [neuroglancer precomputed format] (https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md), based on [this comment](https://github.com/google/neuroglancer/issues/272#issuecomment-752212014). Uses a custom draco exporter based on [this suggestion]( https://github.com/google/neuroglancer/issues/266#issuecomment-739601142).

In the example in the code, input meshes for object id 345809856042 are in directories structured as `test/mito_obj_meshes_s{lod}/345809856042.obj`, where lod is the level of detail.

Running `multiresolution_sharding.py` will read in the meshes and output the precomputed mesh and corresponding json.

![Demo](recording/recording.gif)


