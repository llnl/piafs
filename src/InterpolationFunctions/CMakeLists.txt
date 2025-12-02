add_library(InterpolationFunctions STATIC
  Interp1PrimFifthOrderCRWENO.c
  Interp1PrimFifthOrderCRWENOChar.c
  Interp1PrimFifthOrderCompactUpwind.c
  Interp1PrimFifthOrderCompactUpwindChar.c
  Interp1PrimFifthOrderUpwind.c
  Interp1PrimFifthOrderUpwindChar.c
  Interp1PrimFifthOrderWENO.c
  Interp1PrimFifthOrderWENOChar.c
  Interp1PrimFirstOrderUpwind.c
  Interp1PrimFirstOrderUpwindChar.c
  Interp1PrimFourthOrderCentral.c
  Interp1PrimFourthOrderCentralChar.c
  Interp1PrimSecondOrderMUSCL.c
  Interp1PrimSecondOrderMUSCLChar.c
  Interp1PrimThirdOrderMUSCL.c
  Interp1PrimThirdOrderMUSCLChar.c
  Interp1PrimSecondOrderCentral.c
  Interp1PrimSecondOrderCentralChar.c
  CompactSchemeCleanup.c
  CompactSchemeInitialize.c
  MUSCLInitialize.c
  WENOCleanup.c
  WENOFifthOrderCalculateWeights.c
  WENOFifthOrderInitializeWeights.c
  WENOInitialize.c
)

target_include_directories(InterpolationFunctions PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
