add_library(FirstDerivative STATIC
  FirstDerivativeFirstOrder.c
  FirstDerivativeSecondOrder.c
  FirstDerivativeFourthOrder.c
)

target_include_directories(FirstDerivative PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
