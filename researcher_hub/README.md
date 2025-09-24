## Create an autora-workflow

### Setting up an virtual environment

Install this in an environment using your chosen package manager. In this example we are using virtualenv

Install:

- python (3.8 or greater): https://www.python.org/downloads/
- virtualenv: https://virtualenv.pypa.io/en/latest/installation.html

Install the Prolific Recruitment Manager as part of the autora package:

Change to the directory of the autora_workflow. Here, we define the autora workflow

```shell
cd researcher_environment
```

### Create a virtual environment

```shell
viratualenv venv
```

### Install dependencies

Install the requirements:

```shell
pip install -r requirements.txt
```

### Write your code

The autora_workflow.py file shows a basic example on how to run a closed loop autora experiment. Navigate [here](https://autoresearch.github.io/autora/) for more advanced options.
