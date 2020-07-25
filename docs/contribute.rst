Contribute
==========
**Tonic** is modelled after `PyTorch Vision <https://github.com/pytorch/vision>` to a large extent.
If you want to see a specific dataset or transformation implemented but you're unsure how to do it, please open an issue.
Once you have your code ready on a separate branch, please do the following steps before you open a pull request:

Format the code
~~~~~~~~~~~~~~~~~~
Please use the `black formatter <https://black.readthedocs.io/en/stable/>`_ as a pre-commit hook. You can easily install it as follows:
::

  pip install pre-commit
  pre-commit install

When you use ``git add`` you add files to the current commit, then when you run ``git commit`` the black formatter will run BEFORE the commit itself. If it fails the check, the black formatter will format the file and then present it to you to add it into your commit. Simply run ``git add`` on those files again and do the remainder of the commit as normal.

Run tests
~~~~~~~~~
Whether you add a dataset or transform, please make sure to also add a test for it.
To run the tests, execute the following command from the root directory of the repo:
::

  python -m pytest
