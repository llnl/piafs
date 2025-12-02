add_library(CommonFunctions STATIC
  CommonFunctions.c
)

target_include_directories(CommonFunctions PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
