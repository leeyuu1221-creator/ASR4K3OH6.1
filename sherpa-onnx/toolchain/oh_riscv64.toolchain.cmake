set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_SYSTEM_VERSION 1)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(riscv)")
  message(STATUS "HOST SYSTEM ${CMAKE_HOST_SYSTEM_PROCESSOR}")
  set(CMAKE_C_COMPILER clang)
  set(CMAKE_ASM_COMPILER clang)
  set(CMAKE_CXX_COMPILER clang++)
else()
  if(DEFINED ENV{RISCV_ROOT_PATH})
    file(TO_CMAKE_PATH $ENV{RISCV_ROOT_PATH} RISCV_ROOT_PATH)
  else()
    message(FATAL_ERROR "RISCV_ROOT_PATH env must be defined")
  endif()

  set(RISCV_ROOT_PATH
      ${RISCV_ROOT_PATH}
      CACHE STRING "root path to riscv ohos toolchain")
  set(CMAKE_C_COMPILER "${RISCV_ROOT_PATH}/bin/clang")
  set(CMAKE_ASM_COMPILER "${RISCV_ROOT_PATH}/bin/clang")
  set(CMAKE_CXX_COMPILER "${RISCV_ROOT_PATH}/bin/clang++")
  set(CMAKE_STRIP ${RISCV_ROOT_PATH}/bin/llvm-strip)
  set(CMAKE_FIND_ROOT_PATH ${RISCV_ROOT_PATH})
  set(CMAKE_SYSROOT "${RISCV_ROOT_PATH}/sysroot")
  set(CMAKE_INCLUDE_PATH "${RISCV_ROOT_PATH}/sysroot/usr/include/")
  set(CMAKE_LIBRARY_PATH "${RISCV_ROOT_PATH}/sysroot/usr/lib/")
  set(CMAKE_PROGRAM_PATH "${RISCV_ROOT_PATH}/sysroot/usr/bin/")
  set(CMAKE_CROSSCOMPILING TRUE)
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

add_definitions(-D__OHOS__)

set(STACK_SIZE_BYTES 16777216)

set(RISCV_MARCH_FLAGS
    "-march=rv64gcv_zfh_zvfh_zba_zicbop_zihintpause_xsmtvdotii")

set(CMAKE_C_FLAGS
    "${RISCV_MARCH_FLAGS} -mabi=lp64d -Wno-unused-command-line-argument -fuse-ld=lld -Wl,-z,stack-size=${STACK_SIZE_BYTES} ${CMAKE_C_FLAGS}"
)
set(CMAKE_CXX_FLAGS
    "${RISCV_MARCH_FLAGS} -mabi=lp64d -Wno-unused-command-line-argument -fuse-ld=lld -stdlib=libc++ -static-libstdc++ -Wl,--push-state,-Bstatic -lc++ -lc++abi -Wl,--pop-state -Wl,-z,stack-size=${STACK_SIZE_BYTES} ${CXX_FLAGS}"
)

set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -static-libgcc -static-libstdc++ -Wl,--push-state,-Bstatic -lgcc -lc++ -lc++abi -Wl,--pop-state -lm -Wl,-z,stack-size=${STACK_SIZE_BYTES}"
)

set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -latomic -lm -mabi=lp64d -Wl,-z,stack-size=${STACK_SIZE_BYTES}"
)
set(CMAKE_MODULE_LINKER_FLAGS
    "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-z,stack-size=${STACK_SIZE_BYTES}")
