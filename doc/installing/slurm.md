# Running the OpenQuake Engine on a SLURM cluster

Most supercomputers support a scheduler called SLURM (
Simple Linux Utility for Resource Management). The OpenQuake engine
is able to interface with SLURM to make use of all the resources
of the cluster.

## Running OpenQuake calculations with SLURM

Let's consider a user with ssh access to a SLURM cluster. The only
thing she has to do after logging in is to load the openquake
libraries with the command
```
$ load module openquake
```
Then running a calculation is quite trivial. The user has two choices:
running the calculation on-the-fly with the command
```
$ srun oq engine --run job.ini
```
or running the calculation in a batch with the command
```
$ sbatch oq engine --run job.ini
```
In the first case the engine will log on the console and the progress
of the job as well as the errors, if any, will be visible, so this
is the recommended approach for beginners. In the
second case the progress will not be visible but it
can be extracted from the engine database with
the command
```
$ oq engine --show-log -1
```
where `-1` denotes the last submitted job. Using `sbatch` is recommended
to users that needs to send multiple calculations. Notice at the moment
such calculations will be serialized by the engine queue. This restriction
may be lifted in the future. Even if the jobs are sequential, the subtasks
spawned by them will run in parallel and use all of the machines of the
cluster.

## Running out of quota

Right now the engine store all of its files (intermediate results and
calc_XXX.hdf5 files) under the $HOME/oqdata directory. It is therefore
easy to run out of the quota for large calculations. Fortunaly there
is an environment variable $OQ_DATADIR that can be configured to point
to some other target, like a directory on a large shared disk. Such
directory must be accessible in read/write mode from all workers in
the clusters. Another option is to set a `shared_dir` in the
`openquake.cfg` file and then the engine will store its data under the
path `shared_dir/$HOME/oqdata`. This option is preferable since it will
work transparently for all users but only the sysadmin can set it.

## Installing SLURM

This section is for the administrators of the SLURM cluster.
Installing the engine requires access to PyPI since the universal
installer downloads packages from there. Here are the installations
instructions:

TODO: for Antonio

After installing the engine, the sysadmin has to edit the openquake.cfg
file and set three parameters:
```
[distribution]
oq_distribute = slurm
python = /opt/openquake/bin/python

[dbserver]
host = local
```
The location of the `openquake.cfg` file can be found with the command
```
$ oq info cfg
```
Each user will have its own database located in
$HOME/oqdata/db.sqlite3. The database will be created automatically
the first time the user runs a calculation (NB: in engine-3.18 it must be
created manually with the command `oq engine --upgrade-db`).

## How it works internally

The support for SLURM is implemented in the module
openquake/baselib/parallel.py. The idea is to submit to SLURM a job
array of tasks for each parallel phase of a calculation. For instance
a classical calculations has three phases: preclassical, classical
and postclassical.

The calculation will start sequentially, then it will reach the
preclassical phase: at that moment the engine will create a
bash script called `slurm.sh` and located in the directory
`$HOME/oqdata/calc_XXX` being XXX the calculation ID, which is
an OpenQuake concept and has nothing to do with the SLURM ID.
The `slurm.sh` script has the following template:
```bash
#!/bin/bash
#SBATCH --job-name={mon.operation}
#SBATCH --array=1-{mon.task_no}
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output={mon.calc_dir}/%a.out
#SBATCH --error={mon.calc_dir}/%a.err
srun {python} -m openquake.baselib.slurm {mon.calc_dir} $SLURM_ARRAY_TASK_ID
```
At runtime the `mon.` variables will be replaced with their values:

- `mon.operation` will be the string "preclassical"
- `mon.task_no` will be the total number of tasks to spawn
- `mon.calc_dir` will be the directory $HOME/oqdata/calc_XXX
- `python` will be the path to the python executable to use

System administrators may want to adapt such template. At the moment
this requires modifying the engine codebase; in the future the template
may be moved in the configuration section.

A task in the OpenQuake engine is simply a Python function or
generator taking some arguments and a monitor object (`mon`),
sending results to the submitter process via zmq.

Internally the engine will save the input arguments for each task
in pickle files located in `$HOME/oqdata/calc_XXX/YYY.pik`, where
XXX is the calculation ID and YYY is the SLURM_ARRAY_TASK_ID starting from 1
to the total number of tasks.

The command `srun {python} -m openquake.baselib.slurm {mon.calc_dir}
$SLURM_ARRAY_TASK_ID` in `slurm.sh` will submit the tasks in parallel
by reading the arguments from the input files.

Using a job array has the advantage that all tasks can be killed
with a single command. This is done automatically by the engine
if the user aborts the calculation or if the calculation fails
due to an error.