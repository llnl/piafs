add_library(IOFunctions STATIC
  ReadArray.c
  ReadArraywInterp.c
  WriteArray.c
  WriteBinary.c
  WriteTecplot2D.c
  WriteText.c
)

target_include_directories(IOFunctions PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
