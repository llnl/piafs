Testing
=======

PIAFS includes a comprehensive regression test suite.

Running Tests
-------------

CMake
~~~~~

.. code-block:: bash

   cd build
   make test

   # Or with ctest
   ctest --verbose
   ctest --output-on-failure

Autotools
~~~~~~~~~

.. code-block:: bash

   make check

Test Configuration
------------------

For HPC systems with job schedulers:

**CMake:**

.. code-block:: bash

   cmake -DMPIEXEC="srun" ..
   make test

**Autotools:**

.. code-block:: bash

   ./configure --with-mpiexec="srun"
   make check

Test Suite
----------

The test suite includes:

* 1D Euler equation tests
* 2D Navier-Stokes tests
* 3D Navier-Stokes tests
* Chemistry tests
* MPI parallel tests

Tests compare simulation outputs against benchmark solutions using a relative tolerance of 1.0e-14.

Understanding Results
---------------------

All tests should pass. If tests fail:

1. Check your build configuration
2. Verify MPI is working correctly
3. For GPU builds, ensure GPUs are available
4. Review test logs for specific errors

Test logs are saved in:

* CMake: ``build/Testing/Temporary/``
* Autotools: ``Tests/test_mpi.sh.log``

Adding Tests
------------

To add new tests, see ``Tests/README.md`` and follow the existing test patterns.

Next Steps
----------

* :doc:`../developer/contributing` - Contribute to PIAFS
* :doc:`../examples/index` - Working examples
