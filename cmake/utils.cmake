# cmake/utils.cmake — Ayaka build utilities
# Inspired by vLLM's cmake/utils.cmake

# Python discovery
function(find_python_from_executable PYTHON_EXECUTABLE_PATH SUPPORTED_VERSIONS)
  if(NOT PYTHON_EXECUTABLE_PATH)
    message(FATAL_ERROR
      "Please set AYAKA_PYTHON_EXECUTABLE to the path of the desired python version"
      " before running cmake configure.")
  endif()

  execute_process(
    COMMAND "${PYTHON_EXECUTABLE_PATH}" --version
    OUTPUT_VARIABLE PYTHON_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYTHON_VERSION_RESULT)

  if(NOT PYTHON_VERSION_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to get Python version from ${PYTHON_EXECUTABLE_PATH}")
  endif()

  string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ "${PYTHON_VERSION_OUTPUT}")
  if(NOT CMAKE_MATCH_1 OR NOT CMAKE_MATCH_2)
    message(FATAL_ERROR "Could not parse Python version from: ${PYTHON_VERSION_OUTPUT}")
  endif()
  set(PYTHON_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")

  set(SUPPORTS_VERSION FALSE)
  foreach(SUPPORTED_VERSION IN LISTS SUPPORTED_VERSIONS)
    if(PYTHON_VERSION VERSION_EQUAL SUPPORTED_VERSION)
      set(SUPPORTS_VERSION TRUE)
      break()
    endif()
  endforeach()

  if(NOT SUPPORTS_VERSION)
    message(WARNING
      "Python version ${PYTHON_VERSION} is not in the supported list: ${SUPPORTED_VERSIONS}."
      " Build may still work but is untested.")
  endif()

  message(STATUS "Found Python ${PYTHON_VERSION} at ${PYTHON_EXECUTABLE_PATH}")
endfunction()

# CMake prefix path
# NOTE: Requires Python_EXECUTABLE to be set (via find_package(Python))
function(append_cmake_prefix_path PACKAGE QUERY)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import ${PACKAGE}; print(${QUERY})"
    OUTPUT_VARIABLE PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _result)

  if(_result EQUAL 0 AND PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${PREFIX_PATH}")
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
    message(STATUS "Added ${PACKAGE} cmake prefix path: ${PREFIX_PATH}")
  else()
    message(FATAL_ERROR "${PACKAGE} not found. Install ${PACKAGE} first.")
  endif()
endfunction()

# CUDA architecture management
# Clear default CUDA architecture flags set by torch/cmake
macro(clear_cuda_arches VAR)
  set(${VAR})
  # Remove torch-generated placeholder flags from CMAKE_CUDA_FLAGS
  if(CMAKE_CUDA_FLAGS)
    string(REGEX REPLACE "-gencode arch=compute_[0-9]+,code=sm_[0-9]+ " "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
  endif()
endmacro()
