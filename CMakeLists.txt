cmake_minimum_required(VERSION 3.14)

project(Cpp_DstarLite)

# change these vars with your OWN paths
SET(VAR_PYTHON_PATH "/usr/include/python3.8")
SET(VAR_BOOST_PATH "path-to-your-boost-build/lib/cmake/Boost-1.78.0")

SET(CMAKE_PREFIX_PATH ${VAR_BOOST_PATH})
SET(CMAKE_SHARED_LIBRARY_PREFIX "")

ADD_SUBDIRECTORY(test)

find_package(Boost 1.78.0 EXACT REQUIRED COMPONENTS python)
find_package(PythonLibs 3.8 REQUIRED)
find_package(Boost COMPONENTS numpy REQUIRED)

INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}" ${VAR_PYTHON_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

ADD_LIBRARY(cpp_dstar_lite SHARED cpp_dstar_lite.cpp)
TARGET_LINK_LIBRARIES(cpp_dstar_lite ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

# optional
ADD_EXECUTABLE(main main.cpp)
TARGET_LINK_LIBRARIES(main ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} cpp_dstar_lite)
