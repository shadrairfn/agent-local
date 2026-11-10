#!/usr/bin/env python3
"""
Notebook Converter (convert.py)
Two-way conversion between Jupyter Notebook (.ipynb) and Python script (.py) or Markdown (.md).
Compatible with VS Code Python Interactive, Jupytext, PyCharm, and standard Jupyter.
"""

import sys
import os
import re
import json
import uuid
import argparse
from pathlib import Path

try:
    import nbformat
    from nbformat.validator import normalize as nb_normalize
    HAS_NBFORMAT = True
except ImportError:
    HAS_NBFORMAT = False
    nb_normalize = None


def generate_cell_id() -> str:
    """Generate a unique 8-character hex string for Jupyter cell id (nbformat v4.5+)."""
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Converters: .ipynb -> .py (Percent Format)
# ---------------------------------------------------------------------------

def ipynb_to_py(ipynb_path: Path, output_path: Path = None) -> Path:
    """Converts a Jupyter Notebook (.ipynb) to a Python script (.py) using percent format (# %%)."""
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    chunks = []

    for cell in cells:
        cell_type = cell.get("cell_type", "code")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        if cell_type == "markdown":
            header = "# %% [markdown]\n"
            lines = source.splitlines(True)
            body = "".join(["# " + line if line.strip() else "#\n" for line in lines])
            chunks.append(header + body.rstrip() + "\n")
        elif cell_type == "raw":
            header = "# %% [raw]\n"
            lines = source.splitlines(True)
            body = "".join(["# " + line if line.strip() else "#\n" for line in lines])
            chunks.append(header + body.rstrip() + "\n")
        else:  # code cell
            header = "# %%\n"
            chunks.append(header + source.rstrip() + "\n")

    py_content = "\n".join(chunks).strip() + "\n"

    if output_path is None:
        output_path = ipynb_path.with_suffix(".py")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(py_content)

    return output_path


# ---------------------------------------------------------------------------
# Converters: .py -> .ipynb
# ---------------------------------------------------------------------------

def py_to_ipynb(py_path: Path, output_path: Path = None, kernel_name: str = "python3") -> Path:
    """Converts a Python script (.py) with '# %%' cell markers to a Jupyter Notebook (.ipynb)."""
    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read().replace("\r\n", "\n")

    # Match cell markers: # %%, # %% [markdown], # %% [md], # %% [raw], or # <codecell>
    cell_pattern = re.compile(
        r"^(?:#\s*%%[ \t]*(?:\[([a-zA-Z0-9_-]+)\])?[^\n]*|#\s*<([a-zA-Z0-9_-]+)>[^\n]*)$",
        re.MULTILINE
    )
    matches = list(cell_pattern.finditer(content))

    cells_data = []

    if not matches:
        clean_code = content.strip()
        if clean_code:
            cells_data.append({
                "id": generate_cell_id(),
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in clean_code.splitlines()]
            })
    else:
        first_pos = matches[0].start()
        prefix = content[:first_pos].strip()
        if prefix:
            cells_data.append({
                "id": generate_cell_id(),
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in prefix.splitlines()]
            })

        for i, match in enumerate(matches):
            raw_type = (match.group(1) or match.group(2) or "code").lower()
            start_body = match.end()
            end_body = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[start_body:end_body]

            if raw_type in ("markdown", "md", "markdowncell"):
                lines = body.splitlines()
                uncommented = []
                for line in lines:
                    if line.startswith("# "):
                        uncommented.append(line[2:])
                    elif line in ("#", "#\n"):
                        uncommented.append("")
                    elif line.startswith("#"):
                        uncommented.append(line[1:])
                    else:
                        uncommented.append(line)
                clean_text = "\n".join(uncommented).strip()
                if clean_text:
                    cells_data.append({
                        "id": generate_cell_id(),
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [line + "\n" for line in clean_text.splitlines()]
                    })
            elif raw_type in ("raw", "rawcell"):
                clean_text = body.strip()
                if clean_text:
                    cells_data.append({
                        "id": generate_cell_id(),
                        "cell_type": "raw",
                        "metadata": {},
                        "source": [line + "\n" for line in clean_text.splitlines()]
                    })
            else:  # code
                clean_code = body.strip()
                if clean_code:
                    cells_data.append({
                        "id": generate_cell_id(),
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [line + "\n" for line in clean_code.splitlines()]
                    })

    for cell in cells_data:
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

    nb_dict = {
        "cells": cells_data,
        "metadata": {
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3"
            },
            "kernelspec": {
                "display_name": f"Python 3 ({kernel_name})",
                "language": "python",
                "name": kernel_name
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    if output_path is None:
        output_path = py_path.with_suffix(".ipynb")

    if HAS_NBFORMAT:
        nb = nbformat.from_dict(nb_dict)
        if nb_normalize:
            try:
                nb_normalize(nb)
            except Exception:
                pass
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(nb_dict, f, indent=1, ensure_ascii=False)
            f.write("\n")

    return output_path


# ---------------------------------------------------------------------------
# Converters: .ipynb -> .md & .md -> .ipynb
# ---------------------------------------------------------------------------

def ipynb_to_md(ipynb_path: Path, output_path: Path = None) -> Path:
    """Converts a Jupyter Notebook (.ipynb) to a Markdown (.md) document."""
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    cells = nb_data.get("cells", [])
    chunks = []

    for cell in cells:
        cell_type = cell.get("cell_type", "code")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        if cell_type == "markdown":
            chunks.append(source.strip() + "\n\n")
        elif cell_type == "code":
            chunks.append(f"```python\n{source.strip()}\n```\n\n")
        elif cell_type == "raw":
            chunks.append(f"```text\n{source.strip()}\n```\n\n")

    md_content = "".join(chunks).strip() + "\n"

    if output_path is None:
        output_path = ipynb_path.with_suffix(".md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return output_path


def md_to_ipynb(md_path: Path, output_path: Path = None, kernel_name: str = "python3") -> Path:
    """Converts a Markdown (.md) document with code blocks into a Jupyter Notebook (.ipynb)."""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    code_block_pattern = re.compile(r"^```(?:python|py)?\r?\n(.*?)\r?\n^```", re.MULTILINE | re.DOTALL)
    matches = list(code_block_pattern.finditer(content))

    cells_data = []
    last_end = 0

    for match in matches:
        md_text = content[last_end:match.start()].strip()
        if md_text:
            cells_data.append({
                "id": generate_cell_id(),
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in md_text.splitlines()]
            })

        code_text = match.group(1).strip()
        if code_text:
            cells_data.append({
                "id": generate_cell_id(),
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code_text.splitlines()]
            })

        last_end = match.end()

    remaining_text = content[last_end:].strip()
    if remaining_text:
        cells_data.append({
            "id": generate_cell_id(),
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in remaining_text.splitlines()]
        })

    for cell in cells_data:
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

    nb_dict = {
        "cells": cells_data,
        "metadata": {
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "kernelspec": {"display_name": f"Python 3 ({kernel_name})", "language": "python", "name": kernel_name}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    if output_path is None:
        output_path = md_path.with_suffix(".ipynb")

    if HAS_NBFORMAT:
        nb = nbformat.from_dict(nb_dict)
        if nb_normalize:
            try:
                nb_normalize(nb)
            except Exception:
                pass
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(nb_dict, f, indent=1, ensure_ascii=False)
            f.write("\n")

    return output_path


# ---------------------------------------------------------------------------
# Utility: Clear outputs
# ---------------------------------------------------------------------------

def clear_notebook_outputs(ipynb_path: Path, output_path: Path = None) -> Path:
    """Removes all cell outputs and execution counts to make clean git diffs."""
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    if output_path is None:
        output_path = ipynb_path

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1, ensure_ascii=False)
        f.write("\n")

    return output_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-way conversion between Jupyter Notebooks (.ipynb) and Python (.py) or Markdown (.md)."
    )
    parser.add_argument("input_file", help="Path to the input file (.ipynb, .py, or .md)")
    parser.add_argument("-o", "--output", help="Optional output path")
    parser.add_argument(
        "--to",
        choices=["ipynb", "py", "md"],
        help="Target format (default: auto-detected based on source extension)"
    )
    parser.add_argument(
        "--clear-outputs",
        action="store_true",
        help="Strip outputs and execution counts from .ipynb"
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Kernel name for newly generated .ipynb files (default: python3)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else None
    ext = input_path.suffix.lower()

    if args.clear_outputs and ext == ".ipynb":
        res = clear_notebook_outputs(input_path, out_path)
        print(f"Successfully cleared outputs: {res}")
        return

    target = args.to

    if ext == ".ipynb":
        if target == "md":
            res = ipynb_to_md(input_path, out_path)
            print(f"Converted {input_path.name} -> {res.name} (Markdown)")
        else:
            res = ipynb_to_py(input_path, out_path)
            print(f"Converted {input_path.name} -> {res.name} (Python Script, Percent format)")
    elif ext == ".py":
        res = py_to_ipynb(input_path, out_path, kernel_name=args.kernel)
        print(f"Converted {input_path.name} -> {res.name} (Jupyter Notebook)")
    elif ext == ".md":
        res = md_to_ipynb(input_path, out_path, kernel_name=args.kernel)
        print(f"Converted {input_path.name} -> {res.name} (Jupyter Notebook)")
    else:
        print(f"Error: Unsupported input extension '{ext}'. Expected .ipynb, .py, or .md", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
