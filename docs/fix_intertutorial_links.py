"""Fix inter-tutorial links in Jupyter notebooks.

Fixes all tutorial links when changing the tutorial order/numbering.

Goes through all notebooks and replaces all links of the form
`[Tutorial <number>](tutorial_name.ipynb)` with `[Tutorial <new_number>](tutorial_name.ipynb)`."""

import re
from pathlib import Path
import nbformat

def extract_tutorial_number(nb):
    """
    Search for a markdown cell starting with '# Tutorial <number>:'
    and return the tutorial number as a string.
    """
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            # Look for header lines like "# Tutorial 8:" (ignoring leading spaces)
            m = re.search(r'^\s*#\s*Tutorial\s+(\d+):', cell.source, re.MULTILINE)
            if m:
                return m.group(1)
    return None

def update_markdown_links(nb, mapping, nb_path):
    """
    Replace markdown links with empty link text (e.g. [](ratiometry.ipynb))
    to have the link text be "Tutorial <number>" if the file’s stem appears
    in the mapping.
    """
    intertutorial_pattern = re.compile(r'\[[^\]]*\]\(([^)^/]+\.ipynb)\)')
    other_links_pattern = re.compile(r'\[([^\]]*)\]\((?!(?:[^)/]+\.ipynb))([^)]+)\)')
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            def repl(match):
                target = match.group(1)
                # Use only the file name stem for matching.
                key = Path(target).stem
                if key in mapping:
                    return f"[Tutorial {mapping[key]}]({target})"
                else:
                    return match.group(0)
            cell.source = intertutorial_pattern.sub(repl, cell.source)
            
            for match in other_links_pattern.finditer(cell.source):
                link_text, target = match.group(1), match.group(2)
                # Consider a link "non-external" if it does not start with "http://" or "https://".
                if not (target.startswith("http://") or target.startswith("https://") or target.startswith('#')):
                    print(f"Found unmatched link: [{link_text}]({target}) in file {nb_path.name}")
    return nb

def main():
    tutorials_dir = Path(__file__).parent / "tutorials"

    # First build mapping of tutorial file stem --> tutorial number.
    mapping = {}
    for nb_path in tutorials_dir.glob("*.ipynb"):
        nb = nbformat.read(nb_path, as_version=4)
        tnum = extract_tutorial_number(nb)
        if tnum:
            mapping[nb_path.stem] = tnum
            # print(f"Found tutorial number {tnum}: {nb_path.name}")
        else:
            print(f"No tutorial header in {nb_path.name}")

    # Process each notebook and update link texts.
    for nb_path in tutorials_dir.glob("*.ipynb"):
        nb = nbformat.read(nb_path, as_version=4)
        new_nb = update_markdown_links(nb, mapping, nb_path)
        if new_nb != nb:
            # Only write the notebook back if it has changed.
            nbformat.write(new_nb, nb_path)
            print(f"Updated links in {nb_path.name}")
        # print(f"Processed {nb_path.name}")

if __name__ == "__main__":
    main()