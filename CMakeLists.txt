cmake_minimum_required(VERSION 3.25)
project(FredholmEquationSolver_)

set(CMAKE_CXX_STANDARD 17)

add_executable(FredholmEquationSolver_ main.cpp)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(FredholmEquationSolver_ PRIVATE Eigen3::Eigen)