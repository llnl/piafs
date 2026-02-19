Developer Guide
===============

Information for PIAFS developers and contributors.

.. toctree::
   :maxdepth: 2

   contributing
   code-structure
   adding-physics
   adding-schemes
   gpu-development

Contributing to PIAFS
---------------------

We welcome contributions! See :doc:`contributing` for guidelines.

Development Topics
------------------

:doc:`code-structure`
   Understanding the PIAFS codebase organization

:doc:`adding-physics`
   How to add new physical models

:doc:`adding-schemes`
   Adding new numerical schemes

:doc:`gpu-development`
   Developing GPU kernels

Getting Started
---------------

1. Fork the repository on GitHub
2. Create a feature branch
3. Make your changes following the coding conventions
4. Run the test suite
5. Submit a pull request

See :doc:`contributing` for detailed instructions.

Development Setup
-----------------

Debug Build
~~~~~~~~~~~

.. code-block:: bash

   cmake -DCMAKE_BUILD_TYPE=Debug ..
   make

Testing
~~~~~~~

Always run tests before submitting:

.. code-block:: bash

   make test
   # or
   ctest --verbose

Code Style
----------

* Follow the existing code style
* Add Doxygen comments for new functions
* Include SPDX license headers in new files

Communication
-------------

* GitHub Issues for bug reports and feature requests
* Pull Requests for code contributions
* Email for private inquiries

Next Steps
----------

* :doc:`contributing` - Read the contribution guidelines
* :doc:`code-structure` - Learn the codebase layout
