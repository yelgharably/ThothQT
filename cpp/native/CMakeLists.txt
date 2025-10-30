cmake_minimum_required(VERSION 3.15)
project(cpp_negf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cpp_negf MODULE negf.cpp)

if (WIN32)
  # No "lib" prefix on Windows Python extension
  set_target_properties(cpp_negf PROPERTIES PREFIX "")
endif()

# Uncomment to enable OpenMP
# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#   target_link_libraries(cpp_negf PRIVATE OpenMP::OpenMP_CXX)
# endif()

# Ensure the extension is installed into the wheel's root (top-level import cpp_negf)
install(TARGETS cpp_negf DESTINATION .)
