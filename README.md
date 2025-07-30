# Multiresolution Mesh Creator
![Demo](recording/recording.gif)

This repository is meant to be used to create multiresolution meshes in the [neuroglancer precomputed format](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md), inspired by [this comment](https://github.com/google/neuroglancer/issues/272#issuecomment-752212014) in order to be used with [neuroglancer](https://github.com/google/neuroglancer). It uses [Dask](https://dask.org/) to parallelize the mesh generation, allowing for progress to be monitored via eg. http://localhost:8787/status.

## Installation

Install directly from the GitHub repository using pip:

```bash
pip install git+https://github.com/janelia-cellmap/multiresolution-mesh-creator.git
```

You will now be able to run the code via the command line using `create-multiresolution-meshes`.

## Execution
This code assumes that you have either:
1. A set of high resolution (lod 0) meshes in a single directory, from which you want to create downsampled, lower resolution (higher lod) meshes.
2. Or that you already have meshes generated at all desired lod scales. If so, each set of meshes for a given lod should be in the same directory, named eg. s0, s1, s2 corresponding to lod 0, lod 1, lod 2 etc.

To create the multiresolution mesh format required by neuroglancer, you can run `create-multiresolution-meshes`:
```
$: create-multiresolution-meshes --help
usage: create-multiresolution-meshes [-h] [--num-workers NUM_WORKERS]
                                     config_path

Code to convert single-scale (or a set of multi-scale) meshes to the
neuroglancer multi-resolution mesh format

positional arguments:
  config_path           Path to directory containing run-config.yaml and dask-
                        config.yaml

optional arguments:
  -h, --help            show this help message and exit
  --num-workers NUM_WORKERS, -n NUM_WORKERS
                        Number of workers to launch (i.e. each worker is
                        launched with a single bsub command)
```

`create-multiresolution-meshes` takes in two arguments: a path to a config directory, and the number of desired dask workers.

Example config directories for multiresolution mesh creation of the `test_meshes` can be found in `local-config` and `lsf-config`; these show how you could set run locally or on eg. an LSF cluster. The config directory must contain a `run-config.yaml` file - used to configure the decimation and multiresolution creation, and a `dask-config.yaml` file - used to configure dask itself.

An example `run-config.yaml` file is shown below:
```yaml
required_settings:
        input_path: ../test_meshes           # Path to lod 0 meshes or multiscale meshes
        output_path: ../test_meshes_output   # Path to write out multires meshes
        num_lods: 6                          # Number of levels of detail


optional_decimation_settings:
        box_size: 4                    # lod 0 box size in nm; default is determined based on a guesstimate of a good number of faces per lod 0 chunk
        skip_decimation: False         # Skip mesh decimation if meshes exist; default is false
        decimation_factor: 4           # Factor by which to decimate faces at each lod, ie factor**lod; default is 2
        aggressiveness: 10             # Aggressiveness to be used for decimation; default is 7
        delete_decimated_meshes: True  # Delete decimated meshes, only applied if skip_decimation=False
```
To see all possible setups for `dask-config.yaml`, see [here](https://github.com/dask/dask-jobqueue/blob/main/dask_jobqueue/jobqueue.yaml), where you would comment out all but the type of cluster you plan to run on.

If you already have meshes at all the desired scales in the corresponding `s0`, `s1`,... directories in `input_path` - either because you already ran `create-multiresolution-meshes` or because you already created the meshes elsewhere, then you can set `skip_decimation: True`.

If you only have meshes for lod 0, then you must run mesh decimation. Mesh decimation is performed using the lod 0 meshes, reducing the number of faces by a factor of `decimation_factor**lod`. In the example configs, the decimation factor is 4, so the decimation at each lod is `4**lod`.

NOTE: The actual decimation amount is not guaranteed by the decimation factor, but is instead dependent on other mesh decimation settings as well, such as `aggressiveness`. A lower aggressiveness is more conservative, but means that decimation might not hit its target number of faces. Empirically, `aggressiveness: 10` tends to reach the desired decimation levels, but the default is set to the more conservative value of 7.

## Example
Two meshes are provided in `test_meshes` as an example (a bunny (`1.ply`) and a teapot (`2.ply`)). These are provided at a single, lod 0 scale. To turn these single resolution meshes into neuroglancer formatted multiresolution meshes using 10 local dask workers, you would run the following

```
create-multiresolution-meshes /path/to/local-config -n 10
```

A directory called `/path/to/local-config-{date-time}` will be created which contains dask output as well as an `output.log` file which updates as the run progresses and provides information as to where dask can be monitored. Thus every time you submit a run, a new directory will be created with the date and time appended, within which you can monitor progress. Here are the contents of `output.log`:
```
INFO:2021/11/08 12:16:31: Setup working directory as /path/to/local-config-20211108.121631.
INFO:2021/11/08 12:16:35: Starting dask cluster for decimation...
INFO:2021/11/08 12:16:35: Starting dask cluster for decimation completed in 0.10236787796020508!
INFO:2021/11/08 12:16:35: Check http://127.0.0.1:8787/status for decimation status.
INFO:2021/11/08 12:16:35: Generating decimated meshes...
INFO:2021/11/08 12:16:36: Generating decimated meshes completed in 0.7968666553497314!
INFO:2021/11/08 12:16:40: Starting dask cluster for multires creation...
INFO:2021/11/08 12:16:40: Starting dask cluster for multires creation completed in 0.014244318008422852!
INFO:2021/11/08 12:16:40: Check http://127.0.0.1:8787/status for multires creation status.
INFO:2021/11/08 12:16:40: Generating multires meshes...
INFO:2021/11/08 12:16:43: Completed creation of multiresolution neuroglancer mesh for mesh 2!
INFO:2021/11/08 12:16:45: Completed creation of multiresolution neuroglancer mesh for mesh 1!
INFO:2021/11/08 12:16:45: Generating multires meshes completed in 4.497605323791504!
INFO:2021/11/08 12:16:45: Writing info and segment properties files...
INFO:2021/11/08 12:16:45: Writing info and segment properties files completed in 0.010875463485717773!
INFO:2021/11/08 12:16:45: Deleting decimated meshes...
INFO:2021/11/08 12:16:45: Deleting decimated meshes completed in 0.12245559692382812!
INFO:2021/11/08 12:16:45: Complete! Elapsed time: 13.888124704360962
```

The multiresolution meshes will be in `test_mehes_output/multires/`. You can use something like [http-server](https://www.npmjs.com/package/http-server) to serve up that directory for viewing in neuroglancer. See the gif at the start of this page to see the results of running the above command.

To run this on eg. an LSF cluster with 40 workers, you would do something like this:
```
bsub -n 2 -P your_project_name create-multiresolution-meshes lsf-config -n 40
```

Where the bsub command is used to set up your job to run on the cluster, with eg. 2 cores for the dask driver. `create-multiresolution-meshes` then sets up the dask cluster on the LSF cluster using those drivers.

To run on new meshes, just change the `run-config.yaml` file as desired, and to change dask settings, change `dask-config.yaml` as desired.


