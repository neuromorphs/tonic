Contribute
==========
Of course we welcome contributions. If you want to see a specific dataset implemented but you're unsure how to do it, please open an issue.
When you have your code on a separate branch, please do the following steps before you open a pull request:

Install pre-commit
~~~~~~~~~~~~~~~~~~
Please use the [black formatter](https://black.readthedocs.io/en/stable/) as a pre-commit hook. You can easily install it as follows:
::
  pip install pre-commit
  pre-commit install

When you use ``git add`` you add files to the current commit, then when you run ``git commit`` the black formatter will run BEFORE the commit itself. If it fails the check, the black formatter will format the file and then present it to you to add it into your commit. Simply run ```git add``` on those files again and do the remainder of the commit as normal.

Run tests
~~~~~~~~~
To run the tests, execute the following command from the root directory of the repo:
::
  python -m pytest
