#ifndef CPP_DSTAR_LITE_HPP
#define CPP_DSTAR_LITE_HPP

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <map>
#include <limits>

namespace p = boost::python;
namespace np = boost::python::numpy;

using point_2d = std::pair<int64_t, int64_t>;

double euclidean_dist(const np::ndarray &s1, const np::ndarray &s2) {
    auto ss1 = reinterpret_cast<int64_t *>(s1.get_data());
    auto ss2 = reinterpret_cast<int64_t *>(s2.get_data());
    return std::sqrt(std::pow(ss1[0] - ss2[0], 2) + std::pow(ss1[1] - ss2[1], 2));
}

double euclidean_dist(const point_2d &u, const point_2d &v) {
    return std::sqrt(std::pow(u.first - v.first, 2) + std::pow(u.second - v.second, 2));
}

void raise_val_error(const std::string &msg) {
    PyErr_SetString(PyExc_ValueError, &msg[0]);
    p::throw_error_already_set();
}


struct point2d_hash {
    std::hash<int64_t> _rows = std::hash<int64_t>();
    size_t operator()(const point_2d& point) const {
      return _rows(point.first) | _rows(point.second);
    }
};

struct Element {
    point_2d k;
    double val_1;
    double val_2;

    Element(const np::ndarray &k_, const std::pair<double, double> &vals);
    Element(const point_2d &k_, const std::pair<double, double> &vals);
    Element(const point_2d &k_, double val_1_, double val_2_);

    bool operator==(const Element &rhs) const;
    bool operator!=(const Element &rhs) const;
    bool operator<(const Element &rhs) const;
    bool operator<=(const Element &rhs) const;
    bool operator>(const Element &rhs) const;
    bool operator>=(const Element &rhs) const;

    bool operator==(const point_2d &rhs);
};

class DStarLite {
    point_2d _start;
    point_2d _original_start;
    point_2d _last_update_start;
    point_2d _goal;
    point_2d _original_goal;
    size_t _max_it;
    bool _verbose;
    int64_t _obst_margin;
    int64_t _goal_margin;
    np::ndarray _global_map = np::array(p::make_tuple(), np::dtype::get_builtin<double>());
    std::vector<Element> _queue;
    double _k_m;
    np::ndarray _rhs = np::array(p::make_tuple(), np::dtype::get_builtin<double>());
    np::ndarray _g = np::array(p::make_tuple(), np::dtype::get_builtin<double>());
    size_t _step;
    point_2d _last;
    int64_t _rows, _cols;

    double map_val(const np::ndarray &map, int64_t r, int64_t c) const {
        return reinterpret_cast<double *>(map.get_data())[r * map.shape(1) + c];
    }

    double& map_val(np::ndarray &map, int64_t r, int64_t c) {
        return reinterpret_cast<double *>(map.get_data())[r * map.shape(1) + c];
    }    

    void update_map(const np::ndarray &map) {
        _global_map = map.copy();
        _rows = map.shape(0);
        _cols = map.shape(1);
    }

    auto compute_key(const point_2d &p) const {
        auto g = map_val(_g, p.first, p.second);
        auto rhs = map_val(_rhs, p.first, p.second);
        auto k2 = std::min(g, rhs);
        auto k1 = k2 + euclidean_dist(_start, p) + _k_m;
        return std::make_pair(k1, k2);
    }

    void init(bool restart=false) {
        _k_m = 0.0;
        _step = 0;
        auto data = p::list();
        data.append(std::numeric_limits<double>::infinity());
        data *= (_rows * _cols);
        _g = np::array(data, np::dtype::get_builtin<double>());
        _g = _g.reshape(p::make_tuple(_rows, _cols));
        _rhs = _g.copy();
        map_val(_rhs, _goal.first, _goal.second) = 0.0;
        _queue.clear();
        _queue.emplace_back(_goal, compute_key(_goal));
        if (restart) {
            _original_goal = _goal;
            _last = _start;
            _last_update_start = _start;
        }
    }

    bool near_obstacle(const point_2d &u, int64_t margin = -1) const {
        margin = margin >= 0 ? margin : _obst_margin;
        for (int64_t r = std::max(0L, u.first - margin); r < std::min(_rows, u.first + margin + 1L); r++) {
            for (int64_t c = std::max(0L, u.second - margin); c < std::min(_cols, u.second + margin + 1L); c++) {
                if (std::isinf(map_val(_global_map, r, c))) {
                    return true;
                }
            }
        }
        return false;
    }    

    auto first_avlb(const point_2d &u, const point_2d &v, int64_t margin = -1) const {
        /*
        Search in a square window of increasing side centered on `u` for the nearest point to `v` which is far enough to any obstacle wrt the `margin`.
        */
        margin = margin >= 0 ? margin : _obst_margin;
        int64_t l = std::max(_rows - u.first, _cols - u.second);
        std::map<double, point_2d> valid_pts;
        for (int64_t rad = 1; rad < l - 1; rad++) {
            int64_t u_r[2] = {std::max(0L, u.first - rad), std::min(_rows, u.first + rad)};
            int64_t u_c[2] = {std::max(0L, u.second - rad), (_cols, u.second + rad)};
            // check up and down rows         
            for (int64_t c = u_c[0]; c < u_c[1] + 1; c++) {
                for (int64_t i = 0; i < 2; i++) {
                    if (!near_obstacle(point_2d(u_r[i], c), margin)) {
                        valid_pts[std::abs(u_r[i] - v.first) + std::abs(c - v.second)] = std::make_pair(u_r[i], c);
                    }
                }
            }
            // check left and right columns
            for (int64_t r = u_r[0] + 1; r < u_r[1]; r++) {
                for (int64_t i = 0; i < 2; i++) {
                    if (!near_obstacle(point_2d(r, u_c[i]), margin)) {
                        valid_pts[std::abs(r - v.first) + std::abs(u_c[i] - v.second)] = std::make_pair(r, u_c[i]);
                    }
                }
            }            
            if (!valid_pts.empty()) {
                auto point =  *(valid_pts.begin());
                return point.second;
            }
        }
        // no points available
        return point_2d(-1, -1);
    }

    void move_goal(int64_t margin = -1) {
        std::cout << "Unfeaible map with current goal, relocating...\n";
        margin = margin >= 0 ? margin : _obst_margin;
        auto new_goal = first_avlb(_goal, _last_update_start, margin);
        if (new_goal.first < 0) {
            raise_val_error("Cannot relocate goal on a suitable point");
        }
        else {
            std::cout << "Goal moved from [" << _goal.first << ", " << _goal.second << "] to [" << new_goal.first << ", " << new_goal.second << "]\n";
            _goal = new_goal;
        }
    }

    bool unfeasible() const {
        std::vector<point_2d> points;
        for (int64_t r = std::max(0L, _goal.first - 1L); r < std::min(_rows, _goal.first + 2L); r++) {
            for (int64_t c = std::max(0L, _goal.second - 1L); c < std::min(_cols, _goal.second + 2L); c++) {
                if ((r != _goal.first || c != _goal.second) && !std::isinf(map_val(_global_map, r, c))) {
                    points.emplace_back(r, c);
                }
            }
        }
        for (size_t i = 0; i < points.size(); i++) {
            for (size_t j = 0; j < points.size(); j++) {
                if (i != j && std::abs(points[i].first - points[j].first) + std::abs(points[i].second - points[j].second) == 1) {
                    return false;
                }
            }
        }
        return true;
    }

    auto neighbors(int64_t ux, int64_t uy) const {
        std::vector<point_2d> out;
        for (int64_t r = std::max(0L, ux - 1L); r < std::min(_rows, ux + 2L); r++) {
            for (int64_t c = std::max(0L, uy - 1L); c < std::min(_cols, uy + 2L); c++) {
                if (r != ux || c != uy) {
                    out.emplace_back(r, c);
                }
            }
        }
        return out;
    }

    bool within_margin(const point_2d &u, const point_2d &v, int64_t margin = -1) const {
        margin = margin >= 0 ? margin : _obst_margin;
        return euclidean_dist(u, v) <= margin;
    }

    double cost(const point_2d &u, const point_2d &v) const {
        // bad code here :(
        if (std::isinf(map_val(_global_map, u.first, u.second)) ||
            (std::isinf(map_val(_global_map, v.first, v.second)) && !within_margin(_goal, v, 0)) ||
            (!within_margin(_goal, v) && !within_margin(_start, v) && near_obstacle(v)) ||
            ((within_margin(_goal, v) || within_margin(_start, v)) && near_obstacle(v) && !no_alternative(u, v))
        ) {
            return std::numeric_limits<double>::infinity();
        }
        else {
            return euclidean_dist(u, v);
        }
    }
    
    bool no_alternative(const point_2d &u, const point_2d &v) const {
        /*
        If possible, avoid diagonal moves in front of an obstacle.
        */
        point_2d diff(u.first - v.first, u.second - v.second);
        std::vector<point_2d> diffs;
        if (diff.first == 0) {
            diffs.emplace_back(std::max(0L, v.first - 1L), v.second);
            diffs.emplace_back(std::min(_rows - 1L, v.first + 1L), v.second);
        }
        if (diff.second == 0) {
            diffs.emplace_back(v.first, std::max(0L, v.second - 1L));
            diffs.emplace_back(v.first, std::min(_cols - 1L, v.second + 1L));            
        }
        bool out = true;
        for (size_t i = 0; out && i < diffs.size(); i++) {
            out &= near_obstacle(diffs[i]);
        }
        return out;
    }

    void update_vertex(const point_2d& u) {
        if (u.first != _goal.first || u.second != _goal.second) {
            auto neigh = neighbors(u.first, u.second);
            double min_s = std::numeric_limits<double>::infinity();
            for (const auto &point: neigh) {
                if (cost(u, point) + map_val(_g, point.first, point.second) < min_s) {
                    min_s = cost(u, point) + map_val(_g, point.first, point.second);
                }
            }
            map_val(_rhs, u.first, u.second) = min_s;
        }
        // TODO: linear search here, to be improved. Maybe multiset ?
        auto it = std::find(_queue.begin(), _queue.end(), u);
        if (it != _queue.end()) {
            _queue.erase(it);
            std::make_heap(_queue.begin(), _queue.end(), std::greater<Element>());
        }
        if (map_val(_g, u.first, u.second) != map_val(_rhs, u.first, u.second)) {
            _queue.emplace_back(u, compute_key(u));
            std::push_heap(_queue.begin(), _queue.end(), std::greater<Element>());
        }
    }

    void shortest_path() {
        size_t i = 0;
        while (!_queue.empty() && (_queue.front() < Element(_start, compute_key(_start)) ||
            map_val(_rhs, _start.first, _start.second) != map_val(_g, _start.first, _start.second))) {
                std::pop_heap(_queue.begin(), _queue.end(), std::greater<Element>());
                auto el = _queue.back();
                _queue.pop_back();
                auto u = el.k;
                auto temp = Element(u, compute_key(u));
                if (el < temp) {
                    _queue.emplace_back(temp.k, temp.val_1, temp.val_2);
                    std::make_heap(_queue.begin(), _queue.end(), std::greater<Element>());
                }
                else if (map_val(_g, u.first, u.second) > map_val(_rhs, u.first, u.second)) {
                    map_val(_g, u.first, u.second) = map_val(_rhs, u.first, u.second);
                    auto neigh = neighbors(u.first, u.second);
                    for (const auto &point: neigh) {
                        update_vertex(point);
                    }
                }
                else {
                    map_val(_g, u.first, u.second) = std::numeric_limits<double>::infinity();
                    auto neigh = neighbors(u.first, u.second);
                    neigh.push_back(u);
                    for (const auto &point: neigh) {
                        update_vertex(point);
                    }                    
                }
                i++;
                if (_verbose) {
                    std::cout << "Shortest path step done " << i << "\n";
                }
        }
    }

    void scan(point_2d &last, p::object updated_map) {
        auto ext_map = boost::python::extract<np::ndarray>(updated_map.ptr());
        bool update = ext_map.check();
        if (update) {
            auto _gm = reinterpret_cast<double *>(_global_map.get_data());
            auto map = reinterpret_cast<double *>(ext_map().get_data());
            std::vector<point_2d> changed_pts;
            int64_t c, r;
            auto np_map = ext_map();
            for (int64_t i = 0; i < _cols * _rows; i++) {
                if (_gm[i] != map[i]) {
                    c = i % _cols;
                    r = (i - c) / _rows;
                    changed_pts.emplace_back(r, c);
                }
            }
            update_map(np_map);
            if (unfeasible()) {
                move_goal();
                init(true);
            }
            else {
                _k_m += euclidean_dist(last, _start);
                for (const auto &point: changed_pts) {
                    update_vertex(point);
                }
            }
            last = _start;
            shortest_path();
        }
    }

public:
    np::ndarray start() const;
    void set_start(np::ndarray &start);
    np::ndarray original_start() const;
    np::ndarray last_update_start() const;
    np::ndarray goal() const;
    void set_goal(np::ndarray &goal);
    np::ndarray original_goal() const;
    size_t max_it() const;
    bool verbose() const;
    int64_t obst_margin() const;
    int64_t goal_margin() const;

    DStarLite(
        np::ndarray const &map,
        int64_t  goal_row,
        int64_t  goal_col,
        int64_t start_row,
        int64_t start_col,
        size_t max_it,
        bool verbose,
        int64_t obst_margin,
        int64_t goal_margin
    );

    p::object step(p::object updated_map = p::object());
};


BOOST_PYTHON_MODULE(cpp_dstar_lite)
{
    Py_Initialize();
    np::initialize();
        
    p::class_<DStarLite>("DStarLite", p::init<np::ndarray, int64_t, int64_t, int64_t, int64_t, size_t, bool, int64_t, int64_t>())
        .add_property("start", &DStarLite::start, &DStarLite::set_start)
        .add_property("_original_start", &DStarLite::original_start)
        .add_property("_last_update_start", &DStarLite::last_update_start)
        .add_property("goal", &DStarLite::goal, &DStarLite::set_goal)
        .add_property("_original_goal", &DStarLite::original_goal)
        .add_property("verbose", &DStarLite::verbose)
        .add_property("max_it", &DStarLite::max_it)
        .add_property("obst_margin", &DStarLite::obst_margin)
        .add_property("goal_margin", &DStarLite::goal_margin)
        // .def<void (Foo::*)(A&)>("m1", &Foo::m1)
        // .def<p::object (DStarLite::*)(np::ndarray&)>("step", &DStarLite::step)
        .def<p::object (DStarLite::*)(p::object)>("step", &DStarLite::step, (p::arg("updated_map")=p::object()));
}

#endif