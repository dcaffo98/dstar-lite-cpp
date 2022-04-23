#include "cpp_dstar_lite.hpp"

Element::Element(const np::ndarray &k_, const std::pair<double, double> &vals) {
        k = std::make_pair(reinterpret_cast<int64_t *>(k_.get_data())[0], reinterpret_cast<int64_t *>(k_.get_data())[1]);
        val_1 = vals.first;
        val_2 = vals.second;
}
Element::Element(const point_2d &k_, const std::pair<double, double> &vals): k(k_), val_1(vals.first), val_2(vals.second) {}
Element::Element(const point_2d &k_, double val_1_, double val_2_): k(k_), val_1(val_1_), val_2(val_2_) {}

bool Element::operator==(const Element &rhs) const {return k.first == rhs.k.first && k.second == rhs.k.second;}
bool Element::operator==(const point_2d &rhs) {return k.first == rhs.first && k.second == rhs.second;}
bool Element::operator!=(const Element &rhs) const {return k.first != rhs.k.first || k.second != rhs.k.second;}
bool Element::operator<(const Element &rhs) const {return val_1 < rhs.val_1 && val_2 < rhs.val_2;}
bool Element::operator<=(const Element &rhs) const {return val_1 <= rhs.val_1 && val_2 <= rhs.val_2;}
bool Element::operator>(const Element &rhs) const {return val_1 > rhs.val_1 && val_2 > rhs.val_2;}
bool Element::operator>=(const Element &rhs) const {return val_1 >= rhs.val_1 && val_2 >= rhs.val_2;}


np::ndarray DStarLite::start() const {return np::array(p::make_tuple(_start.first, _start.second), np::dtype::get_builtin<int64_t>());}
void DStarLite::set_start(np::ndarray &start) {_start = point_2d(reinterpret_cast<int64_t *>(start.get_data())[0], reinterpret_cast<int64_t *>(start.get_data())[1]);}
np::ndarray DStarLite::original_start() const {return np::array(p::make_tuple(_original_start.first, _original_start.second), np::dtype::get_builtin<int64_t>());}
np::ndarray DStarLite::last_update_start() const {return np::array(p::make_tuple(_last_update_start.first, _last_update_start.second), np::dtype::get_builtin<int64_t>());}
np::ndarray DStarLite::goal() const {return np::array(p::make_tuple(_goal.first, _goal.second), np::dtype::get_builtin<int64_t>());}
void DStarLite::set_goal(np::ndarray &goal) {_goal = point_2d(reinterpret_cast<int64_t *>(goal.get_data())[0], reinterpret_cast<int64_t *>(goal.get_data())[1]);}
np::ndarray DStarLite::original_goal() const {return np::array(p::make_tuple(_original_goal.first, _original_goal.second), np::dtype::get_builtin<int64_t>());}
size_t DStarLite::max_it() const {return _max_it;}
bool DStarLite::verbose() const {return _verbose;}
int64_t DStarLite::obst_margin() const {return _obst_margin;}
int64_t DStarLite::goal_margin() const {return _goal_margin;}


DStarLite::DStarLite(
        const np::ndarray &map,
        int64_t  goal_row,
        int64_t  goal_col,
        int64_t start_row,
        int64_t start_col,
        size_t max_it,
        bool verbose,
        int64_t obst_margin,
        int64_t goal_margin
    ) {
        _start = point_2d(start_row, start_col);
        _original_start = _start;
        _last_update_start = _start;
        _goal = point_2d(goal_row,  goal_col);
        _original_goal = _goal;
        _max_it = max_it;
        _verbose = verbose;
        _obst_margin = obst_margin;
        _goal_margin = goal_margin;

        if (map.get_dtype() != np::dtype::get_builtin<double>()) {
            PyErr_SetString(PyExc_TypeError, "Incorrect array data type: it must be `double`");
            p::throw_error_already_set();
        }
        if (std::isinf(map_val(map, start_row, start_col))) {
            raise_val_error("Starting point is on an obstacle");
        }
        update_map(map);
        if (unfeasible()) {
            move_goal();
        }
        init();
    }

p::object DStarLite::step(p::object updated_map) {
    auto ext_updated_map = boost::python::extract<np::ndarray>(updated_map.ptr());
    if (ext_updated_map.check() && _step == 0) {
        raise_val_error("Map changed before computing shortest_path at least once");
    }
    auto out = p::object();
    if (_step == 0) {
        _last = _start;
        scan(_last, updated_map);
        shortest_path();
        out = np::array(p::make_tuple(_start.first, _start.second), np::dtype::get_builtin<int64_t>());
    }
    else if (_step < _max_it) {
        point_2d temp(-1, -1);
        scan(_last, updated_map);
        if (euclidean_dist(_goal, _start) > _goal_margin) {
            if (_verbose) {
                std::cout << "Current location: " << "[" << _start.first << ", " << _start.second << "]\n";
            }
            auto neigh = neighbors(_start.first, _start.second);
            double min_s = std::numeric_limits<double>::infinity();
            for (const auto &point: neigh) {
                if (cost(_start, point) + map_val(_g, point.first, point.second) < min_s) {
                    min_s = cost(_start, point) + map_val(_g, point.first, point.second);
                    temp = point;
                }
            }
            if (temp.first < 0 || std::isinf(map_val(_global_map, temp.first, temp.second))) {
                raise_val_error("Cannot find a feasible path");
            }
            _start = temp;
            out = np::array(p::make_tuple(temp.first, temp.second), np::dtype::get_builtin<int64_t>());
        }
    }
    else {
        raise_val_error("Cannot find feasible path within given amount of step");
    }
    _step++;
    return out;
}
