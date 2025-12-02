add_library(BoundaryConditions STATIC
  BCCleanup.c
  BCDirichlet.c
  BCExtrapolate.c
  BCInitialize.c
  BCNoslipWall.c
  BCPeriodic.c
  BCReflect.c
  BCSlipWall.c
  BCSponge.c
  BCSubsonicAmbivalent.c
  BCSubsonicInflow.c
  BCSubsonicOutflow.c
  BCSupersonicInflow.c
  BCSupersonicOutflow.c
)

target_include_directories(BoundaryConditions PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
