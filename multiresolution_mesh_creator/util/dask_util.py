from contextlib import contextmanager
import os
import dask
from dask.distributed import Client
import getpass
import tempfile
import shutil
import multiresolution_mesh_creator.util.io_util as io_util
from datetime import datetime
import yaml
from yaml.loader import SafeLoader


def _set_local_directory(cluster_type):
    # from https://github.com/janelia-flyem/flyemflows/blob/6ae20cb58fd55e74b47a43efdab7e09908a346ba/flyemflows/util/dask_util.py
    # This specifies where dask workers will dump cached data
    local_dir = dask.config.get(f"jobqueue.{cluster_type}.local-directory",
                                None)
    if local_dir:
        return

    user = getpass.getuser()
    local_dir = None
    for d in [f"/scratch/{user}", f"/tmp/{user}"]:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            continue
        else:
            local_dir = d
            dask.config.set(
                {f"jobqueue.{cluster_type}.local-directory": local_dir})

            # Set tempdir, too.
            tempfile.tempdir = local_dir

            # Forked processes will use this for tempfile.tempdir
            os.environ['TMPDIR'] = local_dir
            break

    if local_dir is None:
        raise RuntimeError(
            "Could not create a local-directory in any of the standard places."
        )


@contextmanager
def start_dask(num_workers, msg, logger):

    # Update dask
    with open("dask-config.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)
        dask.config.update(dask.config.config, config)

    cluster_type = next(iter(dask.config.config['jobqueue']))
    _set_local_directory(cluster_type)

    if cluster_type == 'local':
        from dask.distributed import LocalCluster
        cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    else:
        if cluster_type == 'lsf':
            from dask_jobqueue import LSFCluster
            cluster = LSFCluster()
        elif cluster_type == 'slurm':
            from dask_jobqueue import SLURMCluster
            cluster = SLURMCluster()
        elif cluster_type == 'sge':
            from dask_jobqueue import SGECluster
            cluster = SGECluster()
        cluster.scale(num_workers)
    try:
        with io_util.Timing_Messager(f"Starting dask cluster for {msg}",
                                     logger):
            client = Client(cluster)
        io_util.print_with_datetime(
            f"Check {client.cluster.dashboard_link} for {msg} status.", logger)
        yield client
    finally:
        client.shutdown()
        client.close()


def setup_execution_directory(config_path, logger):
    # Create execution dir (copy of template dir) and make it the CWD
    # from flyemflows: https://github.com/janelia-flyem/flyemflows/blob/03cfd79ccc1dcd4903007b36759f4b677ca5c67e/flyemflows/bin/launchflow.py
    timestamp = f'{datetime.now():%Y%m%d.%H%M%S}'
    execution_dir = f'{config_path}-{timestamp}'
    execution_dir = os.path.abspath(execution_dir)
    shutil.copytree(config_path, execution_dir, symlinks=True)
    os.chmod(f'{execution_dir}/run-config.yaml', 0o444)  # read-only
    io_util.print_with_datetime(f"Setup working directory as {execution_dir}.",
                                logger)

    return execution_dir