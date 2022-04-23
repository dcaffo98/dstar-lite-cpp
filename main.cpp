#include "cpp_dstar_lite.hpp"
#include <time.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <array>
#include <algorithm>
#define N 10000

int main(int argc, char *argv[]) {
    Py_Initialize();
    np::initialize();
    int64_t h = 100, w = 100;
    srand(time(NULL));
    double data[N];
    for (size_t i = 0; i < N; i++) {
        auto rnd = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
        data[i] = rnd > 1 ? std::numeric_limits<double>::infinity() : 0;
    }
    for (int r = 0; r < 2; r++) {
        for (int c = 49; c < 52; c++) {
            data[r * w + c] = std::numeric_limits<double>::infinity();
        }
    }
    auto map = np::from_data(data, np::dtype::get_builtin<double>(), p::make_tuple(h, w), p::make_tuple(h * sizeof(double), sizeof(double)), p::object());
    // auto map = np::array(p::make_tuple(0.5, 0.8, 0.3, 0.67), np::dtype::get_builtin<double>());
    // map = map.reshape(p::make_tuple(2, 2));
    DStarLite ds(map, 0, 50, 99, 50 ,1e4, true, 1, 1);
    auto updated_map = map.copy();
    reinterpret_cast<double *>(updated_map.get_data())[99] = 0.5;
    std::vector<point_2d> path;
    auto step = ds.step();
    while (boost::python::extract<np::ndarray>(step).check()) {
        auto s = boost::python::extract<np::ndarray>(step)();
        path.push_back(point_2d(reinterpret_cast<int64_t *>(s.get_data())[0], reinterpret_cast<int64_t *>(s.get_data())[1]));
        if (path.size() > 2 && path.back() == path[path.size() - 3]) {
            break;
        }
        std::cout << "[" << path.back().first << ", " << path.back().second << "]\n";
        step = ds.step();
    }
    np::ndarray x = np::array(p::make_tuple());
    x = np::array(p::make_tuple(1.5, 2.4), np::dtype::get_builtin<double>());
    std::cout << x.shape(0) << " " << ds.original_start().shape(0) << "\n";
    auto a = reinterpret_cast<int64_t *>(ds.start().get_data())[0];
    std::cout << a << "\n";

    return EXIT_SUCCESS;
}