add_library(NavierStokes3D STATIC
  NavierStokes3DCleanup.c
  NavierStokes3DComputeCFL.c
  NavierStokes3DEigen.c
  NavierStokes3DFlux.c
  NavierStokes3DFunctions.c
  NavierStokes3DInitialize.c
  NavierStokes3DParabolicFunction.c
  NavierStokes3DPreStep.c
  NavierStokes3DSource.c
  NavierStokes3DUpwind.c
  NavierStokes3DWriteChem.c
)

target_include_directories(NavierStokes3D PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/physicalmodels
)
