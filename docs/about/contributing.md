# Contributing

If you are new to contributing to open-source projects, please check out [this guide](https://github.com/firstcontributions/first-contributions).

This section assumes you have some familiarity with Git, GitHub, and Python virtualenvs.

## Set Up the Repository

Here are the 3 steps you need to follow to set up the repository:

1. Fork the repository

    Go to [this page](https://github.com/poupeaua/otary) and click the `Fork` button. This will create a copy of the repository in your GitHub account.

    <img width="300px" height="auto" src="https://firstcontributions.github.io/assets/Readme/fork.png" alt="fork this repository"/>

2. Clone the forked repository locally

    Go to your GitHub dashboard, click on the forked repository, and then click the `Code` button. Use the clone method you prefer.

    <img width="300" src="https://firstcontributions.github.io/assets/Readme/clone.png" alt="clone this repository" />

3. Create a new branch

    Once the repository is cloned on your local device, you need to create a new branch.

    Otary follows the [Gitflow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow). You are therefore encourage to create a feature branch from the dev branch.

    Once on the dev branch, create a new branch like this:

    ```bash
    git checkout -b feature/any-name-you-like
    ```

## Install Dependencies

1. Create a virtual environment

    You first need to set up a Python Virtual Environment. You can create it the way you like.

    I would recomment the use of [pyenv](https://github.com/pyenv/pyenv) + [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv). This makes you coding experience smoother when it comes to manage multiple Python versions and multiple virtual environments.

    Activate your virtual environment.

2. Install project dependencies

    This project currently uses Poetry as its dependency manager. If you do not have it installed, you first need to [install Poetry](https://python-poetry.org/docs/#installation).

    !!! info "Poetry environment check"

        You may want to check that Poetry understood you virtual environment correctly and will indeed install all your dependencies in it. You can do a quick check by running:

        ```bash
        poetry env info
        ```

        It should give you information about the virtual environment used by Poetry.

    Once Poetry is installed, run at the root of the project:

    ```bash
    poetry sync
    ```

    For more information about what this command does, please refer to the [Poetry documentation](https://python-poetry.org/docs/cli/#sync).

## Contribute to codebase

You are all done ! You can now start contributing to Otary ! Any idea or suggestion is welcome.

If you do not have any idea, you can start with the ["Good first issue" issues](https://github.com/poupeaua/otary/contribute).

### Run tests

Any change you propose should be tested before submitting a pull request.

The [CI/CD pipeline](https://github.com/poupeaua/otary/actions) already includes tests and checks but controlling code before pushing is always a good idea. Run the following command:

```bash
make full-check
```

This will run all the code quality checks and the tests. Tools used for Otary development are:

- [pytest](https://pypi.org/project/pytest/): the Python testing framework
- [pylint](https://pypi.org/project/pylint/): the basic Python linter
- [ruff](https://pypi.org/project/ruff/): a fast Python linter
- [mypy](https://pypi.org/project/mypy/): a static type checker
- [black](https://pypi.org/project/black/): a code formatter
- [pre-commit](https://pypi.org/project/pre-commit/): a tool to manage git hooks before commiting

For a fine-grained control on checks and tests, you can take a look at the `Makefile` at the root of the repository.

### Interactive Jupyter Notebook development

Since Otary is a image and geometry processing library, **having a visual interface to play with is a must**.

If you need to iterate quickly on the code and see the results right away, you can use a Jupyter notebook.

For this you can put the two following cells at the top of your notebook:

```python
# automatically reload otary after a code change
% load_ext autoreload
% autoreload 2
```

```python
import otary as ot
```

You can put your Jupyter notebooks in the `notebooks/` directory at the root of the repository.

### Principles

Thank you for respecting the following principles when contributing:

- Code tries to reach excellence and follows code best practices. Read the book "Clean Code" by Robert C. Martin if you are interested.
- Keep It Simple, Stupid (KISS principle). Do not overcomplicate things.
- Use type hints
- Function docstrings are written in Google style
- Try not to add new dependencies

About tests:
- Tests are written using pytest and are grouped within classes when possible when they are related
- Try to respect TDD (Test Driven Development) if possible

## Contribute to documentation

The documentation is built using [mkdocs](https://www.mkdocs.org/) and [mkdocs-material](https://squidfunk.github.io/mkdocs-material/).

Start by running the following command:

```bash
make docs-serve
```

The documentation website will be available at `http://127.0.0.1:8000/`

It will be **automatically be updated** when you make any change to files in the `docs/` directory or the `mkdocs.yml` file at the root of the repository. You do not need to re-run the command over and over again after each change.

## Submit a Pull Request

Once you are done with your contribution, you can commit, push and submit a pull request by clicking on the `Compare & pull request` button on the GitHub page on your forked repository.

Thank you for your contribution !
