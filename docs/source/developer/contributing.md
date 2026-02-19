# Contributing to PIAFS

PIAFS is distributed under the terms of the MIT license. All new contributions must be made under this license.

## Getting Started

1. Fork the repository on GitHub
2. Create a feature branch from `master`:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Make your changes following the coding conventions
4. Push your branch and open a pull request against `master`

## Reporting Issues

Use the GitHub issue tracker to report bugs or request features. When reporting a bug, include:

* A concise description of the problem
* The platform, compiler, and MPI library you are using
* Steps to reproduce the issue, ideally with a minimal example
* Any relevant output or error messages

## Submitting Pull Requests

* Keep each pull request focused on a single logical change
* Ensure the code compiles with both CMake and Autotools when possible
* Run the regression test suite and confirm all tests pass
* Add or update tests in `Tests/` if your change affects simulation results
* Update documentation (Doxygen comments and relevant files) to reflect your changes

## Coding Conventions

* Follow the style of the surrounding code
* All new source files must begin with the SPDX license header:
  ```c
  // SPDX-License-Identifier: MIT
  // SPDX-FileCopyrightText: <year>, Lawrence Livermore National Security, LLC
  ```
* Document new functions and data structures with Doxygen-style comments (`/*! ... */`)
* Prefer descriptive variable names and avoid magic numbers
* Define named constants instead of hardcoded values

## Testing

Before submitting a pull request:

```bash
# CMake build
cd build
make test

# Autotools build
make check
```

All tests should pass. If you add new functionality, add corresponding tests.

## Documentation

Update documentation when you make changes:

* Add/update Doxygen comments in header files
* Update relevant documentation in `docs/source/`
* For significant features, consider adding an example in `Examples/`

## Developer Certificate of Origin

By contributing to this project you certify that your contribution was written by you and that you have the right to submit it under the MIT license.

See the [Developer Certificate of Origin](https://developercertificate.org/) for the full text.

## Contact

For questions not suited to a public issue:
* Email: debojyoti.ghosh@gmail.com

## License

LLNL-CODE-2015997

See {doc}`../about/license` for complete license information.
