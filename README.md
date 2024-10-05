# Seismic

2024 NASA Space Apps Challenge Seismic Detection Across the Solar System

# Set up

Import your data into the `data` directory. By default they provide you with directories `lunar` and `mars`, containing testing and training data.

Next, create and source a virtual environment. Here's how to make a virtual environment in the current repo:

First, ensure that you're at the top level directory.

```sh
pwd
```

You should get `*/Seismic`

```sh
python3 -m venv ./venv
```

Then source the directory

```sh
source ./venv/bin/activate
```

Install packages into the virtual environment

```sh
pip install -r requirements.txt
```
