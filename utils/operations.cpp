#include <algorithm>


double set_union(double f, double g) {
    return std::max(f, g);
}


double set_intersection(double f, double g) {
    return std::min(f, g);
}


double set_subtraction(double f, double g) {
    return std::min(f, -g);
}


double set_negation(double f) {
    return -f;
}


