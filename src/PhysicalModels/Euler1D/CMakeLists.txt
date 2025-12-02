add_library(Euler1D STATIC
  Euler1DCleanup.c
  Euler1DComputeCFL.c
  Euler1DEigen.c
  Euler1DFlux.c
  Euler1DFunctions.c
  Euler1DInitialize.c
  Euler1DPreStep.c
  Euler1DSource.c
  Euler1DUpwind.c
  Euler1DWriteChem.c
)

target_include_directories(Euler1D PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/physicalmodels
)
