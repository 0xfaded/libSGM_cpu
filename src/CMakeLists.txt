set(LIBSGM_CPU_SRCS
  "compile_me.cpp")

include_directories(${CMAKE_CURRENT_LIST_DIR})

add_library(libsgm_cpu ${LIBSGM_CPU_SRCS})


add_subdirectory(detail)
