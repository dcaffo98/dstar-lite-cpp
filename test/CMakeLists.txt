cmake_minimum_required(VERSION 3.14)

project(Test)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(Boost 1.78.0 EXACT REQUIRED COMPONENTS python)
find_package(PythonLibs 3.8 REQUIRED)
find_package(Boost COMPONENTS numpy REQUIRED)

INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}" ${VAR_PYTHON_PATH})

enable_testing()

add_executable(
  cpp_dstar_lite_test
  cpp_dstar_lite_test.cpp
)

target_link_libraries(
  cpp_dstar_lite_test
  gtest_main
  cpp_dstar_lite
)

include(GoogleTest)
gtest_discover_tests(cpp_dstar_lite_test)
