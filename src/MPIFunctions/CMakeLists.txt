add_library(MPIFunctions STATIC
  MPIBroadcast.c
  MPICommunicators.c
  MPIExchangeBoundaries1D.c
  MPIExchangeBoundariesnD.c
  MPIGatherArray1D.c
  MPIGatherArraynD.c
  MPIGetArrayDatanD.c
  MPIGetFilename.c
  MPIIOGroups.c
  MPILocalDomainLimits.c
  MPIMax.c
  MPIMin.c
  MPIPartition1D.c
  MPIPartitionArray1D.c
  MPIPartitionArraynD.c
  MPIRank1D.c
  MPIRanknD.c
  MPISum.c
)

target_include_directories(MPIFunctions PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
