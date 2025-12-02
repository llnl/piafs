add_library(LimiterFunctions STATIC
  LimiterGeneralizedMinMod.c
  LimiterMinMod.c
  LimiterSuperBee.c
  LimiterVanLeer.c
)

target_include_directories(LimiterFunctions PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
