#include <gtest/gtest.h>
#include "../cpp_dstar_lite.hpp"
#include <algorithm>

#define H 10
#define W 10
#define START_ROW 0
#define START_COL 4
#define GOAL_ROW 9
#define GOAL_COL 4


class MapTest : public ::testing::Test {
  protected:
    void SetUp() override {
      std::fill_n(&map_[0], H * W, std::numeric_limits<double>::infinity());
    }

    double map_[H * W];

  public:
    double* map() {return map_;}
    double& map(int r, int c) {return map_[r * W + c];}
    void solve(DStarLite &ds, std::vector<point_2d> &path) {
      auto step = ds.step();
      while (boost::python::extract<np::ndarray>(step).check()) {
          auto s = boost::python::extract<np::ndarray>(step)();
          path.push_back(point_2d(reinterpret_cast<int64_t *>(s.get_data())[0], reinterpret_cast<int64_t *>(s.get_data())[1]));
          step = ds.step();
      }
    }
};

TEST_F(MapTest, ShortesPathNoMargin) {
  /*
  ---------------------
  |/|/|/|/|*| | |/|/|/|
  |/|/|/|/|*| | |/|/|/|
  |/|/|/|/|*| | |/|/|/|
  |/|/|/|/|*| | |/|/|/|
  |/|/|/|/|*| | |/|/|/|
  |/|/|/|/|*| | |/|/|/|
  |/|/|/|/|*| | |/|/|/|
  |/|/|/|/|*| | |/|/|/|
  |/|/|/|/|*| | |/|/|/|
  |/|/|/|/|*| | |/|/|/|
  ---------------------
  */
  std::vector<point_2d> expected_path;
  for (int r = 0; r < H; r++) {
    for (int c = START_COL; c < START_COL + 3; c++) {
      map(r, c) = 0;
      if (c == GOAL_COL) {
        expected_path.emplace_back(r, c);
      }
    }
  }
  expected_path.front() = point_2d(START_ROW, START_COL);
  expected_path.back() = point_2d(GOAL_ROW, GOAL_COL);
  ASSERT_EQ(map(9, 5), 0);
  ASSERT_EQ(map(9, 7), std::numeric_limits<double>::infinity());
  auto np_map = np::from_data(map(), np::dtype::get_builtin<double>(), p::make_tuple(H, W), p::make_tuple(H * sizeof(double), sizeof(double)), p::object());
  auto ds = DStarLite(np_map, GOAL_ROW, GOAL_COL, START_ROW, START_COL, 1e4, false, 0, 0);
  std::vector<point_2d> path;
  solve(ds, path);
  ASSERT_EQ(path, expected_path);
}

TEST_F(MapTest, ShortesPathMargin1) {
  /*
  ---------------------
  |/|/|/|/|*| | |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/|*| | |/|/|/|
  ---------------------
  */  
  std::vector<point_2d> expected_path;
  for (int r = 0; r < H; r++) {
    for (int c = START_COL; c < START_COL + 3; c++) {
      map(r, c) = 0;
      if (c == GOAL_COL + 1) {
        expected_path.emplace_back(r, c);
      }
    }
  }
  expected_path.front() = point_2d(START_ROW, START_COL);
  expected_path.back() = point_2d(GOAL_ROW, GOAL_COL);
  ASSERT_EQ(map(9, 5), 0);
  ASSERT_EQ(map(9, 7), std::numeric_limits<double>::infinity());
  auto np_map = np::from_data(map(), np::dtype::get_builtin<double>(), p::make_tuple(H, W), p::make_tuple(H * sizeof(double), sizeof(double)), p::object());
  auto ds = DStarLite(np_map, GOAL_ROW, GOAL_COL, START_ROW, START_COL, 1e4, false, 1, 0);
  std::vector<point_2d> path;
  solve(ds, path);
  ASSERT_EQ(path, expected_path);
}

TEST_F(MapTest, ShortesPathMargin1MapUpdate) {
  /*
  ---------------------
  |/|/|/|/|*| | |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/|*|*| |/|/|/|
  |/|/|/|/|*|/| |/|/|/|
  |/|/|/|/|*| | |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/| |*| |/|/|/|
  |/|/|/|/|*| | |/|/|/|
  ---------------------
  */    
  std::vector<point_2d> expected_path {
    point_2d(START_ROW, START_COL),
    point_2d(1, 5),
    point_2d(2, 5),
    point_2d(3, 5),
    point_2d(3, 4),
    point_2d(4, 4),
    point_2d(5, 4),
    point_2d(6, 5),
    point_2d(7, 5),
    point_2d(8, 5),
    point_2d(GOAL_ROW, GOAL_COL)    
  };
  for (int r = 0; r < H; r++) {
    for (int c = START_COL; c < START_COL + 3; c++) {
      map(r, c) = 0;
    }
  }
  auto np_map = np::from_data(map(), np::dtype::get_builtin<double>(), p::make_tuple(H, W), p::make_tuple(H * sizeof(double), sizeof(double)), p::object());
  auto ds = DStarLite(np_map, GOAL_ROW, GOAL_COL, START_ROW, START_COL, 1e4, false, 1, 0);
  std::vector<point_2d> path;
  auto updated_map = np_map.copy();
  reinterpret_cast<double *>(updated_map.get_data())[4 * W + 5] = std::numeric_limits<double>::infinity();
  auto step = ds.step();
  for (size_t i = 1; boost::python::extract<np::ndarray>(step).check(); i++) {
    auto s = boost::python::extract<np::ndarray>(step)();
    path.push_back(point_2d(reinterpret_cast<int64_t *>(s.get_data())[0], reinterpret_cast<int64_t *>(s.get_data())[1]));
    if (i == 4) {
      step = ds.step(updated_map);
    }
    else {
      step = ds.step();
    }
  }
  ASSERT_EQ(path, expected_path);
}

int main(int argc, char **argv) {
  Py_Initialize();
  np::initialize();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}