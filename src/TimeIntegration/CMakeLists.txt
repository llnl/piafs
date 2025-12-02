add_library(TimeIntegration STATIC
  TimeCleanup.c
  TimeInitialize.c
  TimeExplicitRKCleanup.c
  TimeExplicitRKInitialize.c
  TimePreStep.c
  TimePostStep.c
  TimePrintStep.c
  TimeRHSFunctionExplicit.c
  TimeStep.c
  TimeForwardEuler.c
  TimeRK.c
)

target_include_directories(TimeIntegration PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
