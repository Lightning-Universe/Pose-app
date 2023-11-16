# Contributing

We welcome community contributions to the Lightning Pose App repo! 
If you have found a bug or would like to request a minor change, please 
[open an issue](https://github.com/Lightning-Universe/Pose-app/issues).

In order to contribute code to the repo, please follow the steps below.

We strive to maintain a fun and inclusive environment for our users and contributors.
See our [code of conduct](CODE_OF_CONDUCT.md) for more information.

### Set up a development installation

In order to make changes to Pose App, you will need to [fork](https://guides.github.com/activities/forking/#fork) the
[repo](https://github.com/Lightning-Universe/Pose-app).

If you are not familiar with `git`, check out [this guide](https://guides.github.com/introduction/git-handbook/#basic-git).

Whenever you initially install the Pose App repo, instead of
```bash
pip install -e .
```
run
```bash
pip install -e .[dev]
```
to install additional development tools.

### Create a pull request
After making changes in your fork, 
[open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) 
from your fork. Please read through the rest of this document before submitting the request.

#### Linting
Linters automatically find (and sometimes fix) formatting issues in the code. We use two, which
are run from the command line in the Pose App repo:

* `flake8`: warns of syntax errors, possible bugs, stylistic errors, etc. Please fix these!
```bash
flake8 .
```

* `isort`: automatically sorts import statements
```bash
isort .
```

#### Testing
We currently do not have unit tests or continuous integration (CI) set up for this repo.
For now we will discuss testing of contributions on a case-by-case basis.
