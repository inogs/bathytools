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
and execute the following commands:

```bash
poetry install
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
