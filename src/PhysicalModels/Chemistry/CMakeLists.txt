add_library(Chemistry STATIC
  ChemistryCleanup.c
  ChemistryInitialize.c
  ChemistrySolve.c
  ChemistryWriteSpecies.c
)

target_include_directories(Chemistry PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/physicalmodels
)
