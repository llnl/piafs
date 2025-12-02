# Add all subdirectories
add_subdirectory(ArrayFunctions)
add_subdirectory(BoundaryConditions)
add_subdirectory(CommonFunctions)
add_subdirectory(FirstDerivative)
add_subdirectory(HyParFunctions)
add_subdirectory(InterpolationFunctions)
add_subdirectory(IOFunctions)
add_subdirectory(LimiterFunctions)
add_subdirectory(MathFunctions)
add_subdirectory(MPIFunctions)
add_subdirectory(PhysicalModels)
add_subdirectory(Simulation)
add_subdirectory(TimeIntegration)
add_subdirectory(TridiagLU)

# Main executable
add_executable(PIAFS main.cpp)

# Link all libraries to the main executable
# Order matters for static libraries - dependencies should come after
target_link_libraries(PIAFS
  Simulation
  HyParFunctions
  TimeIntegration
  BoundaryConditions
  Euler1D
  NavierStokes2D
  NavierStokes3D
  Chemistry
  InterpolationFunctions
  LimiterFunctions
  FirstDerivative
  TridiagLU
  IOFunctions
  MPIFunctions
  ArrayFunctions
  MathFunctions
  CommonFunctions
)

# If OpenMP is enabled, link OpenMP library
if(ENABLE_OMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(PIAFS OpenMP::OpenMP_CXX)
  endif()
endif()

# Install the executable
install(TARGETS PIAFS DESTINATION bin)
