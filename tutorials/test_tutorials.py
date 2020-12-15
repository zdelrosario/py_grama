import nbformat
import os
import re
import unittest

from nbconvert.preprocessors import ExecutePreprocessor

# run_notebook() was released by Mike Driscoll under the wxWidgets License
# (c) Mike Driscoll, 2018
# Edits; Zachary del Rosario, 2020
def run_notebook(notebook_path):
    nb_name, _ = os.path.splitext(os.path.basename(notebook_path))
    dirname = os.path.dirname(notebook_path)

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=600, kernel_name="python")
    proc.allow_errors = True

    proc.preprocess(nb, {"metadata": {"path": "/"}})
    output_path = os.path.join(dirname, "{}_all_output.ipynb".format(nb_name))

    with open(output_path, mode="wt") as f:
        nbformat.write(nb, f)
    errors = []
    for cell in nb.cells:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if output.output_type == "error":
                    errors.append(output)

    ## Remove temporary file
    os.remove(output_path)

    return nb, errors


# Find all master notebooks in directory
files_all = os.listdir()
files_notebook = list(filter(lambda s: re.search("ipynb", s) is not None, files_all))
files_master = list(
    filter(lambda s: re.search("master", s) is not None, files_notebook)
)

# Begin test
class TestNotebooks(unittest.TestCase):
    def test(self):
        for filename in files_master:
            print("Testing {}...".format(filename))
            nb, errors = run_notebook(filename)
            self.assertEqual(errors, [])
