Boundary Conditions
===================

PIAFS supports various boundary condition types.

Boundary Condition Types
-------------------------

Periodic
~~~~~~~~

Connects opposite boundaries. Use for problems with periodic domains.

.. code-block:: text

   periodic  0  1
   periodic  0 -1

Extrapolate
~~~~~~~~~~~

Zero-gradient extrapolation (outflow boundary).

.. code-block:: text

   extrapolate  0  1

Dirichlet
~~~~~~~~~

Fixed boundary values. Requires specification of values.

.. code-block:: text

   dirichlet  0  1
   rho  rho*u  rho*v  E

Slip Wall
~~~~~~~~~

Inviscid wall with slip condition.

.. code-block:: text

   slip-wall  1  -1
   u_wall  v_wall

Noslip Wall
~~~~~~~~~~~

Viscous no-slip wall.

.. code-block:: text

   noslip-wall  1  -1
   u_wall  v_wall

Subsonic Inflow/Outflow
~~~~~~~~~~~~~~~~~~~~~~~

Characteristic-based subsonic boundaries.

.. code-block:: text

   subsonic-inflow   0  1
   rho  u  v

   subsonic-outflow  0  -1
   p

Supersonic Inflow/Outflow
~~~~~~~~~~~~~~~~~~~~~~~~~

Supersonic boundaries.

.. code-block:: text

   supersonic-inflow   0  1
   rho  u  v  p

   supersonic-outflow  0  -1

See :doc:`input-files` for complete boundary specification format.
