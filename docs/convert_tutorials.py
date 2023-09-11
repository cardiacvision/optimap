import jupytext
import jupytext.config
from pathlib import Path

for path in Path(".").glob("*.ipynb"):
    nb = jupytext.read(path)

    remove_cells_for_tags = ["remove-input"]

    def f(cell):
        for tag in remove_cells_for_tags:
            if tag in cell.metadata.get("tags", []):
                return False
        return True
    nb['cells'] = list(filter(f, nb.cells))

    config = jupytext.config.JupytextConfiguration()
    config.notebook_metadata_filter = "-all"
    config.cell_metadata_filter = "-all"

    jupytext.write(nb, path.with_suffix('.py'), fmt="py:percent", config=config)