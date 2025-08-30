cmake_minimum_required(VERSION 3.16...3.22)

project(
  frantik
  VERSION 0.1
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VAMP_ARCH} -Wall -Wextra -Wno-ignored-attributes")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 ${VAMP_FAST_ARGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VAMP_ARCH} -Wall -Wextra -Wno-ignored-attributes")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 ${VAMP_FAST_ARGS}")

file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.1/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=117cbf2711572f113bab262933eb5187b08cfc06dce0714a1ee94f2183ddc3ec
)
set(CPM_USE_LOCAL_PACKAGES ON)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
CPMAddPackage("gh:wjakob/nanobind#9a25aed8a7edfe60ef9ad1c911e57667bc4916c4")

nanobind_add_module(_core_ext
  NOMINSIZE
  src/cpp/frantik.cc
)

target_link_libraries(_core_ext
  PRIVATE
  Eigen3::Eigen
)

if($ENV{GITHUB_ACTIONS})
  set(STUB_PREFIX "")
else()
  set(STUB_PREFIX "${CMAKE_BINARY_DIR}/stubs/")
endif()

nanobind_add_stub(
  frantik_stub
  MODULE _core_ext
  OUTPUT "${STUB_PREFIX}__init__.pyi"
  PYTHON_PATH $<TARGET_FILE_DIR:_core_ext>
  DEPENDS _core_ext
  VERBOSE
)

install(
  TARGETS _core_ext
  LIBRARY
  DESTINATION frantik/_core
)

install(
  FILES "${STUB_PREFIX}__init__.pyi"
  DESTINATION "${CMAKE_SOURCE_DIR}/src/frantik/_core"
)
