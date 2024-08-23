SUBDIRS = \
  ArrayFunctions \
  BoundaryConditions \
  CommonFunctions \
  FirstDerivative \
  HyParFunctions \
  InterpolationFunctions \
  IOFunctions \
  LimiterFunctions \
  PhysicalModels \
  MathFunctions \
  MPIFunctions \
  Simulation \
  TimeIntegration \
  TridiagLU

bin_PROGRAMS = PIAFS2D
PIAFS2D_SOURCES = main.cpp
PIAFS2D_LDADD = \
  Simulation/libSimulation.a \
  HyParFunctions/libHyParFunctions.a \
  TimeIntegration/libTimeIntegration.a \
  BoundaryConditions/libBoundaryConditions.a \
  PhysicalModels/Euler1D/libEuler1D.a \
  PhysicalModels/NavierStokes2D/libNavierStokes2D.a \
  InterpolationFunctions/libInterpolationFunctions.a \
  LimiterFunctions/libLimiterFunctions.a \
  FirstDerivative/libFirstDerivative.a \
  TridiagLU/libTridiagLU.a \
  IOFunctions/libIOFunctions.a \
  MPIFunctions/libMPIFunctions.a \
  ArrayFunctions/libArrayFunctions.a \
  MathFunctions/libMathFunctions.a \
  CommonFunctions/libCommonFunctions.a
