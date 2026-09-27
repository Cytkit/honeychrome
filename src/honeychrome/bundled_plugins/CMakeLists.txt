# CMakeLists.txt — AutoSpectral Optimization kernel (pybind11 + Armadillo)
#
# Prerequisites (macOS, local + CI):
#   brew install armadillo libomp cmake
#   pip install pybind11
# Prerequisites (Windows, CI): Armadillo + OpenBLAS via vcpkg (static, see
#   the BLA_STATIC block below), pybind11 via pip — see build.yml for the
#   exact CI step.
# Linux is not an active CI target today (see build.yml matrix — the ubuntu
# runners are commented out) — no apt-based prerequisites have been set up
# or verified for this extension yet.
#
# Usage: run build_autospectral_opt_kernel.py rather than invoking cmake
# directly — it wires up Python3_EXECUTABLE and the OpenMP option for you.

cmake_minimum_required(VERSION 3.18)
project(autospectral_opt_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_dir_result
)
if(NOT _pybind11_dir_result EQUAL 0)
  message(FATAL_ERROR
    "Could not run '${Python3_EXECUTABLE} -m pybind11 --cmakedir' — "
    "is pybind11 installed for this interpreter? "
    "Try: ${Python3_EXECUTABLE} -m pip install pybind11 --break-system-packages")
endif()

# Static OpenBLAS on Windows/Linux — single binary, nothing extra to track through the PyInstaller
# spec or the codesign/notarization pass. macOS is unaffected either way:
# Armadillo resolves to Apple's Accelerate framework, already on-system, so
# this flag has no effect there.
if(NOT APPLE)
  set(BLA_STATIC ON)
endif()

find_package(pybind11 REQUIRED)
find_package(Armadillo REQUIRED)

pybind11_add_module(_autospectral_opt_kernel
  autospectral_opt_pybind.cpp
  core_joint_unmix.cpp
)

target_include_directories(_autospectral_opt_kernel PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(_autospectral_opt_kernel PRIVATE ${ARMADILLO_LIBRARIES})

option(HONEYCHROME_OPENMP "Enable OpenMP in the AutoSpectral Optimization kernel" ON)

if(HONEYCHROME_OPENMP)
  if(APPLE)
    # Homebrew libomp detection, same approach as build_som_kernel.py.
    execute_process(
      COMMAND brew --prefix libomp
      OUTPUT_VARIABLE LIBOMP_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _libomp_prefix_result
    )
    if(_libomp_prefix_result EQUAL 0 AND LIBOMP_PREFIX)
      target_compile_options(_autospectral_opt_kernel PRIVATE
        -Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include)
      target_link_directories(_autospectral_opt_kernel PRIVATE ${LIBOMP_PREFIX}/lib)
      target_link_libraries(_autospectral_opt_kernel PRIVATE omp)
      message(STATUS "AutoSpectral Optimization kernel: macOS OpenMP via libomp at ${LIBOMP_PREFIX}")
    else()
      message(WARNING
        "HONEYCHROME_OPENMP=ON but 'brew --prefix libomp' failed — "
        "building single-threaded. Run: brew install libomp")
      set(HONEYCHROME_OPENMP OFF)
    endif()
  else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(_autospectral_opt_kernel PRIVATE OpenMP::OpenMP_CXX)
  endif()
else()
  message(STATUS "AutoSpectral Optimization kernel: building single-threaded (HONEYCHROME_OPENMP=OFF).")
endif()
