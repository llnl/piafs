add_library(MathFunctions STATIC
  FillGhostCells.c
  FindInterval.c
  InterpolateGlobalnDVar.c
)

target_include_directories(MathFunctions PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
