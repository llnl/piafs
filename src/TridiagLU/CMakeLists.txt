add_library(TridiagLU STATIC
  blocktridiagLU.c
  blocktridiagIterJacobi.c
  tridiagLU.c
  tridiagLUGS.c
  tridiagIterJacobi.c
  tridiagLUInit.c
)

target_include_directories(TridiagLU PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
