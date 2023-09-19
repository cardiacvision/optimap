```{toctree}
:maxdepth: 2
:hidden:
:caption: 🚀 Getting Started

chapters/getting_started.md
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 🚀 Tutorials

tutorials/basics.ipynb
tutorials/motion_compensation.ipynb
tutorials/ratiometry.ipynb
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Notes
chapters/contributing
chapters/misc
chapters/bibliography
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 📚 API Reference

api/optimap
```

% Generate API reference
```{eval-rst}
.. autosummary::
   :toctree: api
   :template: custom-module-template.rst
   :recursive:
   :hidden:

   optimap
```

```{include} ../README.md
```