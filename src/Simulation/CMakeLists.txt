add_library(Simulation STATIC
  Cleanup.c
  EnsembleSimulationsDefine.cpp
  Initialize.c
  InitializeBoundaries.c
  InitializePhysics.c
  InitializePhysicsData.c
  InitializeSolvers.c
  InitialSolution.c
  OutputSolution.cpp
  ReadInputs.c
  SimulationWriteErrors.c
  SingleSimulationDefine.cpp
  Solve.cpp
  WriteInputs.c
)

target_include_directories(Simulation PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
