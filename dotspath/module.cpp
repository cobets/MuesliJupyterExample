#define PY_SSIZE_T_CLEAN
#include <set>
#include <list>
#include <vector>
#include <algorithm>
#include <iterator>
#include <Python.h>

struct Point {
    int x;
    int y;
    Point(int x0, int y0) : x(x0), y(y0) {}
    friend bool operator < (const Point& left, const Point& right);
    friend bool operator == (const Point& left, const Point& right);
};

bool operator < (const Point& left, const Point& right)
{
    return left.x * 10000 + left.y < right.x * 10000 + right.y;
}

bool operator == (const Point& left, const Point& right)
{
    return (left.x == right.x) && (left.y == right.y);
}

// https://www.algorithms-and-technologies.com/point_in_polygon/c
bool point_in_polygon(std::vector<Point> polygon, struct Point point) {
    //A point is in a polygon if a line from the point to infinity crosses the polygon an odd number of times
    bool odd = false;
    // int totalCrosses = 0; // this is just used for debugging
    //For each edge (In this case for each point of the polygon and the previous one)
    for (int i = 0, j = polygon.size() - 1; i < polygon.size(); i++) { // Starting with the edge from the last to the first node
        //If a line from the point into infinity crosses this edge
        if (((polygon[i].y > point.y) != (polygon[j].y > point.y)) // One point needs to be above, one below our y coordinate
            // ...and the edge doesn't cross our Y coordinate before our x coordinate (but between our x coordinate and infinity)
            && (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x)) {
            // Invert odd
            // System.out.println("Point crosses edge " + (j + 1));
            // totalCrosses++;
            odd = !odd;
        }
        //else {System.out.println("Point does not cross edge " + (j + 1));}
        j = i;
    }
    // System.out.println("Total number of crossings: " + totalCrosses);
    //If the number of crossings was odd, the point is in the polygon
    return odd;
}

std::set<Point> get_dot_neighbours(Point v) {
    std::set<Point> result;
    result.insert(Point(v.x - 1, v.y));
    result.insert(Point(v.x - 1, v.y - 1));
    result.insert(Point(v.x, v.y - 1));
    result.insert(Point(v.x + 1, v.y - 1));
    result.insert(Point(v.x + 1, v.y));
    result.insert(Point(v.x + 1, v.y + 1));
    result.insert(Point(v.x, v.y + 1));
    result.insert(Point(v.x - 1, v.y + 1));
    return result;
}

static void do_find_paths(
    std::set<Point> trace_player, 
    std::set<Point> trace_opponent, 
    Point v,
    std::list<std::list<Point>>* paths,
    std::list<Point> path,
    std::set<Point>* path_set
) {
    path.push_back(v);
    path_set->insert(v);
    std::set<Point> dot_neighbours = get_dot_neighbours(v);

    std::list<Point> neighbours;
    std::set_intersection(
        dot_neighbours.begin(), dot_neighbours.end(),
        trace_player.begin(), trace_player.end(),
        std::back_inserter(neighbours)
    );

    for (auto const& nv : neighbours) {
        if (path.front() == nv) {
            for (auto const& opponent : trace_opponent) {
                std::vector<Point> vec { std::begin(path), std::end(path) };
                if (point_in_polygon(vec, opponent)) {
                    paths->push_back(path);
                    break;
                }
            }
        }
        else {
            // C++20 if (!path_set->contains(nv)) {
            if (path_set->find(nv) == path_set->end()) {
                std::list<Point> path_copy;
                std::copy(path.begin(), path.end(), std::back_inserter(path_copy));
                do_find_paths(trace_player, trace_opponent, nv, paths, path_copy, path_set);
            }
        }
    }
}

/*
    args: trace_player, trace_opponent, x, y
*/
static PyObject* find_paths(PyObject* self, PyObject* args) {
    PyObject* trace_player;
    PyObject* trace_opponent;
    int vx, vy;
    if (PyArg_ParseTuple(args, "OOii", &trace_player, &trace_opponent, &vx, &vy)) {
        std::set<Point> trace_player_set;
        std::set<Point> trace_opponent_set;
        PyObject* iter;
        PyObject* item;
        iter = PyObject_GetIter(trace_player);
        while (item = PyIter_Next(iter)) {
            PyObject* xo = PyNumber_Long(PyTuple_GET_ITEM(item, 0));
            PyObject* yo = PyNumber_Long(PyTuple_GET_ITEM(item, 1));
            int x = PyLong_AsLong(xo);
            int y = PyLong_AsLong(yo);
            Py_DECREF(item);
            Py_DECREF(xo);
            Py_DECREF(yo);
            trace_player_set.insert(Point(x, y));
        }
        Py_DECREF(iter);
        
        iter = PyObject_GetIter(trace_opponent);
        while (item = PyIter_Next(iter)) {
            PyObject* xo = PyNumber_Long(PyTuple_GET_ITEM(item, 0));
            PyObject* yo = PyNumber_Long(PyTuple_GET_ITEM(item, 1));
            int x = PyLong_AsLong(xo);
            int y = PyLong_AsLong(yo);
            Py_DECREF(item);
            Py_DECREF(xo);
            Py_DECREF(yo);
            trace_opponent_set.insert(Point(x, y));
        }
        Py_DECREF(iter);
        // Py_DECREF(trace_player);
        // Py_DECREF(trace_opponent);

        std::list<std::list<Point>> paths;
        std::list<Point> path;
        std::set<Point> path_set;
        do_find_paths(
            trace_player_set,
            trace_opponent_set,
            Point(vx, vy),
            &paths,
            path,
            &path_set
        );

        PyObject* result = PyList_New(0);
        for (auto const& p : paths) {
            PyObject* pp = PyList_New(0);
            for (auto const& e : p) {
                PyObject* v = Py_BuildValue("ii", e.x, e.y);
                PyList_Append(pp, v);
            }
            PyList_Append(result, pp);
        }

        return result;
    }
}

static PyMethodDef dotspath_methods[] = {
	{"find_paths", (PyCFunction)find_paths, METH_VARARGS, NULL},
	{NULL,  NULL, 0, NULL}
};

static PyModuleDef dotspath_module = {
	PyModuleDef_HEAD_INIT,
	"dotspath",
	"game dots path routine",
	-1,
	dotspath_methods
};

PyMODINIT_FUNC PyInit_dotspath(void) {
	return PyModule_Create(&dotspath_module);
}
