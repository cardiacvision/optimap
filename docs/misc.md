(opencv)=
# OpenCV with CUDA support

TODO: add instructions for installing OpenCV with CUDA support

(development)=
# Development

To install optimap in development mode use:
```bash
git clone git@github.com:cardiacvision/optimap.git
cd optimap
pip install -e .[all]
```
optimap can be updated by running `git pull` in the `optimap` directory. Changes to the Python code will be reflected immediately.

```{note}
optimap contains C++ extensions that require a C++ compiler to be installed. On Windows, we recommend installing [Visual Studio](https://visualstudio.microsoft.com/vs/) with the C++ development tools.

Changes to the C++ code require recompiling the extensions by running `pip install -e .` again.
```

## Testing
optimap uses [pytest](https://docs.pytest.org/en/stable/) for unit testing. Install pytest with ```pip install pytest``` and run the tests using:
```bash
python -m pytest
```

## Documentation

Run
```bash
pip install .[docs]
```
to install the documentation dependencies. The documentation can then be built using
```bash
python -m sphinx -b html docs docs/_build/html
```
and viewed by opening `docs/_build/html/index.html` in a web browser.

# Bibliography

```{bibliography}
```