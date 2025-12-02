add_library(HyParFunctions STATIC
  ApplyBoundaryConditions.c
  BoundaryIntegral.c
  CalculateError.c
  CalculateConservationError.c
  ExactSolution.c
  HyperbolicFunction.c
  IncrementFilename.c
  NonLinearInterpolation.c
  SourceFunction.c
  VolumeIntegral.c
)

target_include_directories(HyParFunctions PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
