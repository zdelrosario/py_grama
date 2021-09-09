# Contributing to `py_grama`

Thanks for your interest in contributing! Please follow the guidelines below, and contact one of the Maintainers (listed below) if you have any questions.

## Reporting an issue

If you find a issue with the software, please file a [new issue](https://github.com/zdelrosario/py_grama/issues). Please include a [reproducible example](https://stackoverflow.com/help/minimal-reproducible-example) in your issue. Note that if you have a question, please don't file an issue---the issue tracker is meant to document issues in the design and implementation of `py_grama`, not to answer questions.

## Contributing to the software

We welcome contributions to `py_grama`! To contribute, please determine what sort of contribution you plan to make. For more detailed information on forking and branching in the context of contributing, please see [this guide](https://opensource.com/article/19/7/create-pull-request-github).

### Bug-fix

If you find an issue with the software or want to fix an existing issue, please follow these steps:

1. If one does not yet exist, please report an Issue following the instructions in **Reporting an issue** above.
2. Fork `py_grama` and clone a local copy of the repository for your work.
3. Create a branch for your fix with the name `fix_name`, where `name` should be sensibly related to your fix.
4. If one does not already exist, create a unittest in the [tests](https://github.com/zdelrosario/py_grama/tree/master/tests) that captures the bug.
5. Implement your fix.
6. Verify the fix against the test suite; the `Makefile` in `py_grama` automates testing with the spell `make test`.
7. Create a pull request against `py_grama`; one of the Maintainers will review your contribution.

### Feature addition

If you wish to add a new feature to `py_grama`, please follow these steps:

1. Fork `py_grama` and clone a local copy of the repository for your work.
2. Create a branch for your fix with the name `dev_name`, where `name` should be sensibly related to your fix.
3. Create a an appropriate set of tests in the [tests](https://github.com/zdelrosario/py_grama/tree/master/tests) that verify the functionality of your feature.
4. Implement your feature.
5. Verify the feature against the test suite; the `Makefile` in `py_grama` automates testing with the spell `make test`.
6. Create a pull request against `py_grama`; one of the Maintainers will review your contribution.

### Design change

If you have a suggestion for a significant change to the design of `py_grama`, please reach out directly to one of the Maintainers (listed below).

## List of Maintainers

Updated 2020-06-05

- Zachary del Rosario (zdelrosario(at)outlook(doot)com)
