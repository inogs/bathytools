# Bathytools

Bathytools is a suite of tools designed to create a suitable bathymetry
file for use with the MITgcm model, enabling the discretization of a
marine domain.

This software downloads bathymetric datasets, trims the data based on
boundaries specified by the user, and interpolates the dataset onto the cells
of the user-requested domain grid.

Additionally, Bathytools allows users to perform various operations on the
bathymetry before saving it to disk. These operations, referred to as `Actions`,
provide flexibility to modify certain aspects of the bathymetry. This can help
generate smoother meshes or make specific adjustments, such as removing a lagoon,
thereby improving the quality of simulations.

---

### Running the Code

To get started, clone the repository from GitHub by executing the following
command:

```bash
git clone git@github.com:inogs/bathytools.git
```

This software uses `poetry` as its package manager. If you have `poetry`
installed, navigate into the `bathytools` directory created during cloning,
and execute the following command:
```bash
poetry install
```

If you are a developer (i.e., you think that you may need to modify the code),
it is convenient to install this package together with the dev tools. In this
case, instead of a simple "install", you may execute the following commands:

```bash
poetry install --with=dev
poetry run pre-commit install
```

These commands instruct `poetry` to set up a virtual environment
containing all the dependencies required to run `bathytools`.

Once the installation is complete, you can run the software by executing:

```bash
poetry run bathytools
```

---

### Running Without Poetry

If you do not have `poetry` installed, you have two options:

1. Install `poetry`. This is generally straightforward on most systems.
   For OGS users working on the G100 cluster, refer to the section below for
   a detailed guide on installing `poetry` in your environment.

2. Attempt to run `bathytools` in your current Python environment. However,
   this method is less reliable as it requires ensuring that all dependencies
   are already installed. Specifically, you need to have the
   [`bit.sea`](https://github.com/inogs/bit.sea) library installed.
   `bit.sea` is a Python tool designed for scientific oceanographic applications
   and is essential for preprocessing and postprocessing tasks in computational
   workflows involving oceanographic models. Beside `bit.sea`, `bathytools`
   suite uses only very standard Python packages that you should be able to find
   in every common setup: `NumPy`, `Xarray` and `PyYAML`.

---

### Installing Poetry on G100

If you are a user of G100, the computing cluster managed by CINECA, you can
follow the steps below to create a virtual environment with `poetry` installed.

1. **Load a Suitable Python Interpreter**
   Begin by loading the appropriate Python module:
   ```bash
   module load python/3.11.7--gcc--10.2.0
   ```

2. **Choose a Location for the Virtual Environment**
   Decide where you want to store your virtual environment. This location does
   not need to exist beforehand. Set the chosen path as an environment variable:
   ```bash
   export VENV_PATH=$HOME/poetry_env
   ```

3. **Create the Virtual Environment**
   Use the following command to create the virtual environment:
   ```bash
   python3 -m venv $VENV_PATH
   ```

4. **Activate the Virtual Environment**
   Activate the virtual environment by running:
   ```bash
   source $VENV_PATH/bin/activate
   ```

   If the activation is successful, you should see the name of your virtual
   environment appear in front of your shell prompt.

5. **Install Poetry**
   Now you can install `poetry` within the virtual environment using:
   ```bash
   pip install poetry
   ```

6. **Follow Instructions to Install Bathytools**
   Once `poetry` is successfully installed, you can proceed with the eartlier
   instructions in this document to install `bathytools`.

**Important Note:**
When you log in again, make sure to re-load the Python module and activate the virtual environment to continue using `poetry`. To do so, repeat the following two commands:
```bash
module load python/3.11.7--gcc--10.2.0
source $VENV_PATH/bin/activate
```
