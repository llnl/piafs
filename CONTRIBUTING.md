# Contributing to PIAFS

PIAFS is distributed under the terms of the MIT license. All new contributions
must be made under this license. See [License.md](License.md) and
[NOTICE.md](NOTICE.md) for details.

LLNL-CODE-2015997

## Getting Started

1. Fork the repository on GitHub.
2. Create a feature branch from `master`:
   ```
   git checkout -b feature/my-feature
   ```
3. Make your changes, following the coding conventions below.
4. Push your branch and open a pull request against `master`.

## Reporting Issues

Please use the GitHub issue tracker to report bugs or request features. When
reporting a bug, include:

- A concise description of the problem.
- The platform, compiler, and MPI library you are using.
- Steps to reproduce the issue, ideally with a minimal example from the
  `Examples/` directory.
- Any relevant output or error messages.

## Submitting Pull Requests

- Keep each pull request focused on a single logical change.
- Ensure the code compiles cleanly with both the CMake and Autotools build
  systems when possible.
- Run the regression test suite (`make test` or `make check`) and confirm all
  tests pass before submitting.
- Add or update tests in `Tests/` if your change affects simulation results.
- Update documentation (Doxygen comments and relevant `doc/` files) to reflect
  your changes.

## Coding Conventions

- Follow the style of the surrounding code.
- All new source files must begin with the SPDX license header:
  ```c
  // SPDX-License-Identifier: MIT
  // SPDX-FileCopyrightText: <year>, Lawrence Livermore National Security, LLC
  ```
- Document new functions and data structures with Doxygen-style comments
  (`/*! ... */`).
- Prefer descriptive variable names and avoid magic numbers; define named
  constants instead.

## Developer Certificate of Origin

By contributing to this project you certify that your contribution was written
by you and that you have the right to submit it under the MIT license. See the
[Developer Certificate of Origin](https://developercertificate.org/) for the
full text.

## Contact

For questions not suited to a public issue, contact the maintainer at
[debojyoti.ghosh@gmail.com](mailto:debojyoti.ghosh@gmail.com).
