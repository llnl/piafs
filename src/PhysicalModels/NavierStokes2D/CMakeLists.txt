add_library(NavierStokes2D STATIC
  NavierStokes2DCleanup.c
  NavierStokes2DComputeCFL.c
  NavierStokes2DEigen.c
  NavierStokes2DFlux.c
  NavierStokes2DFunctions.c
  NavierStokes2DInitialize.c
  NavierStokes2DParabolicFunction.c
  NavierStokes2DPreStep.c
  NavierStokes2DSource.c
  NavierStokes2DUpwind.c
  NavierStokes2DWriteChem.c
)

target_include_directories(NavierStokes2D PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/physicalmodels
)
