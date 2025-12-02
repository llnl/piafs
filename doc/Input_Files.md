Input Files {#Input_Files}
===========

The following input files are required to run HyPar. They are indicated as being mandatory or 
optional. If optional, then default values are used if not found.

\b Note: It is best to start with the provided examples, understand their input files based on the 
information on this page, and then modify them to run other cases.

\section solver_inp solver.inp

Requirement: \b mandatory

Read by: ReadInputs()

Description: Specify the main simulation-related parameters.

Format: ASCII text

        begin
            <keyword>   <value>
            <keyword>   <value>
            <keyword>   <value>
            ...
            <keyword>   <value>
        end

where the list of keywords and their type are:\n
Keyword name       | Type         | Variable                      | Default value
------------------ | ------------ | ----------------------------- | ------------------------
ndims              | int          | #HyPar::ndims                 | 1
nvars              | int          | #HyPar::nvars                 | 1
size               | int[ndims]   | #HyPar::dim_global            | must be specified
ghost              | int          | #HyPar::ghosts                | 1
n_iter             | int          | #HyPar::n_iter                | 0
restart_iter       | int          | #HyPar::restart_iter          | 0
time_scheme        | char[]       | #HyPar::time_scheme           | euler
time_scheme_type   | char[]       | #HyPar::time_scheme_type      | none
hyp_space_scheme   | char[]       | #HyPar::spatial_scheme_hyp    | 1
hyp_interp_type    | char[]       | #HyPar::interp_type           | characteristic
par_space_scheme   | char[]       | #HyPar::spatial_scheme_par    | 2
dt                 | double       | #HyPar::dt                    | 0.0
conservation_check | char[]       | #HyPar::ConservationCheck     | no
screen_op_iter     | int          | #HyPar::screen_op_iter        | 1
file_op_iter       | int          | #HyPar::file_op_iter          | 1000
op_file_format     | char[]       | #HyPar::op_file_format        | text
ip_file_type       | char[]       | #HyPar::ip_file_type          | ascii
input_mode         | char[]       | #HyPar::input_mode            | serial
output_mode        | char[]       | #HyPar::output_mode           | serial
op_overwrite       | char[]       | #HyPar::op_overwrite          | no
model              | char[]       | #HyPar::model                 | must be specified
size_exact         | int[ndims]   | #HyPar::dim_global_ex         | #HyPar::dim_global

\b Notes:
+ "ndims" \b must be specified \b before "size" and "iproc".
+ if "input_mode" or "output_mode" are set to "parallel" or "mpi-io",
  the number of I/O ranks must be specified right after as an integer.
  For example:

      begin
          ...
          input_mode  parallel 4
          ...
      end

  This means that 4 MPI ranks will participate in file I/O (assuming
  total MPI ranks is more than 4) (see ReadArrayParallel(), 
  WriteArrayParallel(), ReadArrayMPI_IO() ).
  - The number of I/O ranks specified for "input_mode" and "output_mode"
    \b must \b be \b same. Otherwise, the value for the one specified last
    will be used.
  - The number of I/O ranks must be such that the total number of MPI ranks
    is an integer multiple. Otherwise, the code will use only 1 I/O rank.
+ If any of the keywords are not present, the default value is used, except
  the ones whose default values say "must be specified". Thus, keywords that
  are not required for a particular simulation may be left out of the 
  solver.inp input file. For example, 
  - a #Euler1D simulation does not need "par_space_type" or "par_space_scheme"
    because it does not have a parabolic term.
  - unless a conservation check is required, "conservation_check" can be left
    out and the code will not check for conservation.
  - "immersed_body" need not be specified if there are no immersed bodies present.
    \b NOTE: However, if it is specified, and a file of that filename does not
    exist, it will result in an error.

\section boundary_inp boundary.inp

Requirement: \b mandatory

Read by: InitializeBoundaries()

Description: Specify boundary conditions.

Format: ASCII text
        
        nb
        boundary_type   spatial_dimension   face   [extent] 
        boundary_type   spatial_dimension   face   [extent] 
        ...
        boundary_type   spatial_dimension   face   [extent] 

where
+ \b nb is the number of boundaries
+ This is followed by at least \b nb rows, each row specifying:
  - \b boundary_type - Type of boundary, eg. #_PERIODIC_, #_EXTRAPOLATE_, #_DIRICHLET_, etc. See include/boundaryconditions.h for a full list.
  - \b spatial_dimension - The spatial dimension along which this boundary condition applies. Must be in the range 0, 1, ..., #HyPar::ndims-1
  - \b face - The face on which this the boundary acts (1 means the left face along the dimension, or the low index end, -1 means the right face
              along the dimension, or the high index end).
  - [\b extent] - The extent of this boundary (in terms of coordinates, not grid indices). Written as (2 X #HyPar::ndims) real numbers:
                  xmin_0  xmax_0  xmin_1  xmax_1 ... ... xmin_(ndims-1)  xmax_(ndims-1), where [xmin_i, xmax_i] define the extent along
                  spatial dimension i. Note that the xmin_i and xmax_i for i = spatial_dimension are ignored, but should be there for the
                  sake of a uniform format (they can be set as 0).
+ For the following boundary types, an additional line is required right below the line specifying that boundary with inputs specific to that 
  boundary type:
  - #_DIRICHLET_: the next line should specify the boundary value of the solution at that boundary (the value is assumed to be constant in space
    and time). Each component of the solution must be specified (i.e. #HyPar::nvars real numbers).

        u[0]  u[1]  ... u[#HyPar::nvars-1]

  - #_SLIP_WALL_, #_NOSLIP_WALL_: the next line should specify the wall velocity (assumed to be constant 
    in space and time). Each component of the velocity must be specified (i.e. #HyPar::ndims real numbers).

        u[0]  u[1]  ... u[#HyPar::ndims-1]

  - #_SUBSONIC_INFLOW_: the next line must specify the inflow density and velocity vector (assumed to be constant in space and time). Each component
    of the velocity must be specified (i.e. #HyPar::ndims+1 real numbers)

        rho  u[0]  u[1]  ... u[#HyPar::ndims-1]

  - #_SUBSONIC_OUTFLOW_: the next line must specify the outflow pressure (assumed to be constant in space and time), i.e., one real number

        p

  - #_SUPERSONIC_INFLOW_: the next line must specify the inflow density, velocity, and pressure (assumed to be constant in space and time). Each component
    of the velocity must be specified (i.e. #HyPar::ndims+2 real numbers)

        rho  u[0]  u[1]  ... u[#HyPar::ndims-1]  p

    Note that this boundary can also be implemented through the #_DIRICHLET_; however, the flow variables must be converted to the conserved variable
    form to specify the Dirichlet boundary value:

        rho  rho*u[0]  rho*u[1]  ... rho*u[#HyPar::ndims-1]  E (internal energy)

\section initial_inp initial.inp

Requirement: \b mandatory

Read by: InitialSolution(), through ReadArray()

Format: Depends on #HyPar::input_mode and #HyPar::ip_file_type, specified in \b solver.inp. See ReadArraySerial(), ReadArrayParallel(), and 
        ReadArrayMPI_IO() to understand the format and data layout of this file.

Description: This file contains the initial solution. It will probably not be created by hand. See the examples provided for codes that generate
             this file for various problems. The final part in these codes, where the initial solution is written to file, can be reused to 
             generate this file for a new case.

\section lusolver_inp lusolver.inp

Requirement: \b optional

Read by: tridiagLUInit()

Description: Specify parameters related to LU solvers for tridiagonal systems of equations. This file is relevant only if
             some method requiring the solution of a tridiagonal system of equations is being used (for example:
             Interp1PrimFifthOrderCRWENO() ).

Format: ASCII text
        
        begin
            <keyword>   <value>
            <keyword>   <value>
            <keyword>   <value>
            ...
            <keyword>   <value>
        end

where the list of keywords are:

Keyword name       | Type         | Variable                      | Default value
------------------ | ------------ | ----------------------------- | ------------------------
evaluate_norm      | int          | #TridiagLU::evaluate_norm     | 1
maxiter            | int          | #TridiagLU::maxiter           | 10
atol               | double       | #TridiagLU::atol              | 1e-12
rtol               | double       | #TridiagLU::rtol              | 1e-10
verbose            | int          | #TridiagLU::verbose           | 0
reducedsolvetype   | char[]       | #TridiagLU::reducedsolvetype  | #_TRIDIAG_JACOBI_


\section weno_inp weno.inp

Requirement: \b optional

Read by: WENOInitialize() 

Description: Specify parameters related to WENO-type spatial discretization. This file is relevant only if a WENO-type 
             method is being used for spatial discretization (for example: Interp1PrimFifthOrderCRWENO(), Interp1PrimFifthOrderWENO(),
             Interp1PrimFifthOrderHCWENO() ). For most cases, this file is useful if a very specific study on the behavior of the
             WENO-type method is being carried out; typically, the default values are enough to ensure "good" solutions.

Format: ASCII text
        
        begin
            <keyword>   <value>
            <keyword>   <value>
            ...
            <keyword>   <value>
        end

where the list of keywords are:

Keyword name       | Type         | Variable                      | Default value
------------------ | ------------ | ----------------------------- | ------------------------
mapped             | int          | #WENOParameters::mapped       | 0
borges             | int          | #WENOParameters::borges       | 0
yc                 | int          | #WENOParameters::yc           | 0
no_limiting        | int          | #WENOParameters::no_limiting  | 0
epsilon            | double       | #WENOParameters::eps          | 1e-6
rc                 | double       | #WENOParameters::rc           | 0.3
xi                 | double       | #WENOParameters::xi           | 0.001
tol                | double       | #WENOParameters::tol          | 1e-16

\section physics_inp physics.inp

Requirement: \b mandatory/optional (depends on the physical model)

Read by: The initialization function of the various physics modules (eg. Euler1DInitialize(), NavierStokes2DInitialize(), etc)

Description: This file contains inputs specific to a physics model. Depending on the physical model being used, it may or may not
             be mandatory. The exact parameters to specify depend on the physics.

Format: ASCII text

        begin
            <keyword>   <value>
            <keyword>   <value>
            ...
            <keyword>   <value>
        end

The list of keywords depends on the physical model being used:

\subsection physics_inp_euler1d Euler1D

Read by: Euler1DInitialize()

Keyword name       | Type         | Variable                      | Default value
------------------ | ------------ | ----------------------------- | ------------------------
gamma              | double       | #Euler1D::gamma               | 1.4
upwinding          | char[]       | #Euler1D::upw_choice          | "roe" (#_ROE_)
include_chemistry  | char[]       | #Euler1D::include_chem        | "no"

\b Note: "physics.inp" is \b optional; if absent, default values will be used.

\subsection physics_inp_navierstokes2d NavierStokes2D

Read by: NavierStokes2DInitialize()

Keyword name       | Type         | Variable                        | Default value
------------------ | ------------ | ------------------------------- | ------------------------
gamma              | double       | #NavierStokes2D::gamma          | 1.4
Pr                 | double       | #NavierStokes2D::Pr             | 0.72
Re                 | double       | #NavierStokes2D::Re             | -1
T_ref              | double       | #NavierStokes2D::Tref           | 273.15 (Kelvin)
T_0                | double       | #NavierStokes2D::T0             | 275.0  (Kelvin)
T_S                | double       | #NavierStokes2D::TS             | 110.4  (Kelvin)
T_A                | double       | #NavierStokes2D::TA             | 245.4  (Kelvin)
T_B                | double       | #NavierStokes2D::TB             |  27.6  (Kelvin)
upwinding          | char[]       | #NavierStokes2D::upw_choice     | "roe" (#_ROE_)
include_chemistry  | char[]       | #NavierStokes2D::include_chem   | "no"
write_output       | char[]       | #NavierStokes2D::write_op       | "yes"

\b Note: "physics.inp" is \b optional; if absent, default values will be used.

\subsection physics_inp_navierstokes3d NavierStokes3D

Read by: NavierStokes3DInitialize()

Keyword name       | Type         | Variable                        | Default value
------------------ | ------------ | ------------------------------- | ------------------------
gamma              | double       | #NavierStokes3D::gamma          | 1.4
Pr                 | double       | #NavierStokes3D::Pr             | 0.72
Re                 | double       | #NavierStokes3D::Re             | -1
T_ref              | double       | #NavierStokes3D::Tref           | 273.15 (Kelvin)
T_0                | double       | #NavierStokes3D::T0             | 275.0  (Kelvin)
T_S                | double       | #NavierStokes3D::TS             | 110.4  (Kelvin)
T_A                | double       | #NavierStokes3D::TA             | 245.4  (Kelvin)
T_B                | double       | #NavierStokes3D::TB             |  27.6  (Kelvin)
upwinding          | char[]       | #NavierStokes3D::upw_choice     | "roe" (#_ROE_)
include_chemistry  | char[]       | #NavierStokes3D::include_chem   | "no"
write_output       | char[]       | #NavierStokes3D::write_op       | "yes"

\b Note: "physics.inp" is \b optional; if absent, default values will be used.

\section simulation_inp simulation.inp

Requirement: \b mandatory if running an ensemble/multidomain simulation

Read by: EnsembleSimulation::define()

Description: Specify the parameters related to ensemble simulations.

Format: ASCII text

        begin
            <keyword>   <value>
            <keyword>   <value>
            <keyword>   <value>
            ...
            <keyword>   <value>
        end

where the list of keywords and their type are:\n
Keyword name       | Type         | Variable                                      | Default value
------------------ | ------------ | --------------------------------------------- | ------------------------
nsims              | int          | #EnsembleSimulation::m_nsims                  | 1

\b Notes:
+ This file \b must exist for ensemble simulations;
  if this file does not exist, HyPar will run a standard single simulation.

\section chemistry_inp chemistry.inp

Requirement: \b optional

Read by: ChemistryInitialize()

Description: Specify parameters related to the chemistry module (photochemistry).

Format: ASCII text

        begin
            <keyword>   <value>
            <keyword>   <value>
            <keyword>   <value>
            ...
            <keyword>   <value>
        end

where the list of keywords and their type are:\n
Keyword name       | Type         | Variable                        | Default value
------------------ | ------------ | ------------------------------- | ------------------------
lambda_UV          | double       | #Chemistry::lambda_UV           | 2.48e-7 (248 nm)
theta              | double       | #Chemistry::theta               | 0.17*pi/180 radians
f_CO2              | double       | #Chemistry::f_CO2               | 0
f_O3               | double       | #Chemistry::f_O3                | 0.005
Ptot               | double       | #Chemistry::Ptot                | 101325 Pa
Ti                 | double       | #Chemistry::Ti                  | 288 K
Lz                 | double       | #Chemistry::Lz                  | 0.03 (30 mm)
z_mm               | double       | #Chemistry::z_mm                | 0
nz                 | int          | #Chemistry::nz                  | 20
t_start            | double       | #Chemistry::t_start             | 0.0 s
t_pulse            | double       | #Chemistry::t_pulse             | 1e-8 s (10 nanoseconds)
k0a                | double       | #Chemistry::k0a                 | 0.9*3.3e-13 s^{-1}
k0b                | double       | #Chemistry::k0a                 | 0.1*3.3e-13 s^{-1}
k1a                | double       | #Chemistry::k1a                 | 0.8*3.95e-13 s^{-1}
k1b                | double       | #Chemistry::k1a                 | 0.2*3.95e-13 s^{-1}
k2a                | double       | #Chemistry::k2a                 | 1.2e-16 s^{-1}
k2b                | double       | #Chemistry::k2a                 | 1.2e-16 s^{-1}
k3a                | double       | #Chemistry::k3a                 | 1.2e-17 s^{-1}
k3b                | double       | #Chemistry::k3a                 | 1.0e-17 s^{-1}
k4                 | double       | #Chemistry::k4                  | 1.1e-16 s^{-1}
k5                 | double       | #Chemistry::k5                  | 2.0e-16 s^{-1}
k6                 | double       | #Chemistry::k6                  | 0.2*3.0e-17 s^{-1}
F0                 | double       | #Chemistry::F0                  | 2000 J/m^2
sO3                | double       | #Chemistry::sO3                 | 1.1e-21 m^2
IA                 | double       | #Chemistry::IA                  | 1.0
IB                 | double       | #Chemistry::IB                  | 1.0
IC                 | double       | #Chemistry::IC                  | 0.0
write_all_zlocs    | double       | #Chemistry::write_all_zlocs     | "yes"

\b Note: "chemistry.inp" is \b optional; if absent, default values will be used.
