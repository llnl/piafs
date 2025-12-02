add_library(ArrayFunctions STATIC
  ArrayImplementations.c
)

target_include_directories(ArrayFunctions PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
