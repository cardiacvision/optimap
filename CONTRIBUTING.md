# Contributing

We welcome and encourage everyone to contribute to optimap!

Contributions can be questions, bug reports, feature requests and new code.
Here is how to get started.

## Development

To install optimap in development mode use:

```bash
git clone git@github.com:cardiacvision/optimap.git
cd optimap
pip install -e .[all]
```

optimap can be updated by running `git pull` in the `optimap` directory. Changes to the Python code will be reflected immediately.

```{note}
optimap contains C++ extensions that require a C++ compiler to be installed. On Windows, we recommend installing [Visual Studio](https://visualstudio.microsoft.com/vs/) with the C++ development tools.
```

```{warning}
Changes to the C++ code require recompiling the extensions by running `pip install -e .` again. If you see changes to the C++ code during a `git pull`, please run the `pip install` command again.
```

## Issues

### Questions and Feature Requests

We encourage all users to submit questions and ideas for improvements to the optimap project.
For this please create an issue on the [issue page](https://github.com/cardiacvision/optimap/issues).

### Reporting Bugs

For bug reports, please make sure that you are using the latest version of optimap by running

```bash
pip install --upgrade opticalmapping[all]
```

and checking if the bug still exists. Whenever possible, please provide error messages, sample code, screenshots or other files when [creating a new issue](https://github.com/cardiacvision/optimap/issues/new).

## Contributing New Code

Any code contributions are welcome, whether fixing a typo or bug, adding new functionality, improve core functionality, or anything that you think should be in the repository.

Please open a [pull request](https://github.com/cardiacvision/optimap/pulls) and link to an open issue if applicable.

### Testing

optimap uses [pytest](https://docs.pytest.org/en/stable/) for unit testing. Install pytest with `pip install pytest` and run the tests using:

```bash
python -m pytest
```

All tests are located in the `test` folder. We encourage you to include test code in a pull request.

### Documentation

To build and preview the documentation, you need to install the documentation dependencies:

```bash
pip install .[docs]
```

The documentation can then be built using

```bash
python -m sphinx -b html docs docs/_build/html
```

and viewed by opening `docs/_build/html/index.html` in a web browser.

We use reST syntax for API documentation in the python code, while the documentation text is written in MyST Markdown flavour. See the [MyST syntax cheat sheet](https://jupyterbook.org/en/stable/reference/cheatsheet.html) for instructions on using MyST.
