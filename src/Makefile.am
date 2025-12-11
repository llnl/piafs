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

bin_PROGRAMS = PIAFS@PIAFS_NAME_SUFFIX@
PIAFS@PIAFS_NAME_SUFFIX@_SOURCES = main.cpp
PIAFS@PIAFS_NAME_SUFFIX@_LDADD = \
  Simulation/libSimulation.a \
  HyParFunctions/libHyParFunctions.a \
  TimeIntegration/libTimeIntegration.a \
  BoundaryConditions/libBoundaryConditions.a \
  PhysicalModels/Euler1D/libEuler1D.a \
  PhysicalModels/NavierStokes2D/libNavierStokes2D.a \
  PhysicalModels/NavierStokes3D/libNavierStokes3D.a \
  PhysicalModels/Chemistry/libChemistry.a \
  InterpolationFunctions/libInterpolationFunctions.a \
  LimiterFunctions/libLimiterFunctions.a \
  FirstDerivative/libFirstDerivative.a \
  TridiagLU/libTridiagLU.a \
  IOFunctions/libIOFunctions.a \
  MPIFunctions/libMPIFunctions.a \
  ArrayFunctions/libArrayFunctions.a \
  MathFunctions/libMathFunctions.a \
  CommonFunctions/libCommonFunctions.a
