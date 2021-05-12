Contribute
==========
**Tonic** is modelled after `PyTorch Vision <https://github.com/pytorch/vision>` to a large extent.
If you want to see a specific dataset or transformation implemented but you're unsure how to do it, please open an issue on Github.
Once you have your code ready on a separate branch, please do the following steps before you open a pull request:

Test it
~~~~~~~~~
Whether you add a dataset or transform, please make sure to also add a test for it.
Especially the functional tests for transforms are needed to make sure that transformations do what they are supposed to do.
To add your functional test, have a look at the tests already there under `tonic/test`.
To run the tests, execute the following command from the root directory of the repo:
::

  python -m pytest

If you are working on something to try to make a specific test pass, you can also add a filter to save time:
::

  python -m pytest -k my_test_name
