#ifndef PRIMITIVES_H
#define PRIMITIVES_H

double
primitive_plane(double x, double y, double z, const double parameters[4]);
double
primitive_sphere(double x, double y, double z, const double parameters[4]);
double
primitive_cylinder(double x, double y, double z, const double parameters[7]);
double
primitive_torus(double x, double y, double z, const double parameters[8]);
double
primitive_cone(double x, double y, double z, const double parameters[7]);
double
primitive_ellipsoid(double x, double y, double z, const double parameters[9]);

#endif
