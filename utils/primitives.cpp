#include <algorithm>
//#include <vector>
#include <cmath>
#include <cassert>


static double compute_dot_product(double v1[], double v2[]) {
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}


static double compute_norm2(double v[]) {
  return sqrt(compute_dot_product(v, v));
}


// Signed distance to an infinite plane defined by the given parameters.
// Negative in the normal direction; positive in the opposite direction.
double primitive_plane(double x, double y, double z, 
                       const double parameters[4]) 
{
  //assert(parameters.size() == 4);
  double normalvec[] = {parameters[0], parameters[1], parameters[2]};
  double dist = parameters[3];
  double p[] = {x,y,z};
  double d = compute_dot_product(normalvec, p) - dist;
  return -d;
}


// Signed distance to a sphere defined by the given parameters.
// Positive inside; negative outside
double primitive_sphere(double x, double y, double z, 
                        const double parameters[4])
{
  //assert(parameters.size() == 4);
  double center[] = {parameters[0], parameters[1], parameters[2]};
  double radius = parameters[3];
  double X = center[0] - x;
  double Y = center[1] - y;
  double Z = center[2] - z;
  double d = sqrt(X*X + Y*Y + Z*Z) - radius;
  return -d;
}


double primitive_cylinder(double x, double y, double z, 
                          const double parameters[7]) 
{
  //assert(parameters.size() == 7);
  double axis_dir[] = {parameters[0], parameters[1], parameters[2]};
  double axis_pos[] = {parameters[3], parameters[4], parameters[5]};
  double radius = parameters[6];
  double diff[] = {x-axis_pos[0], y-axis_pos[1], z-axis_pos[2]};
  double lamb = compute_dot_product(axis_dir, diff);
  double v[] = {diff[0] - lamb*axis_dir[0], diff[1] - lamb*axis_dir[1],
                diff[2] - lamb*axis_dir[2]};
  double axis_dist = compute_norm2(v);
  double d = axis_dist - radius;
  return -d;
}


double primitive_torus(double x, double y, double z, 
                       const double parameters[8]) 
{
  //assert(parameters.size() == 8);
  double normalvec[] = {parameters[0], parameters[1], parameters[2]};
  double center[] = {parameters[3], parameters[4], parameters[5]};
  double rminor = parameters[6];
  double rmajor = parameters[7];
  double s[] = {x-center[0], y-center[1], z-center[2]};
  double spin1 = compute_dot_product(normalvec, s);
  double spin0vec[] = {s[0] - spin1*normalvec[0], 
                       s[1] - spin1*normalvec[1],
                       s[2] - spin1*normalvec[2]};
  double spin0 = compute_norm2(spin0vec) - rmajor;
  double d = sqrt(spin0*spin0 + spin1*spin1) - rminor;
  return -d;
}


double primitive_cone(double x, double y, double z, 
                      const double parameters[7]) 
{
  //assert(parameters.size() == 7);
  double axis_dir[] = {parameters[0], parameters[1], parameters[2]};
  double center[] = {parameters[3], parameters[4], parameters[5]};
  double angle = parameters[6];
    
  double s[] = {x-center[0], y-center[1], z-center[2]};
  double g = compute_dot_product(s, axis_dir);
  double slen = compute_norm2(s);
  double sqrs = slen*slen;
  double f = sqrs - g*g;
    
  f = std::max(f, 0.0);
  f = sqrt(f);

  double da = cos(angle) * f;
  double db = -sin(angle) * g;
    
  double d;
    
  if (g<0.0 && (da-db)<0.0) {
    d = sqrt(sqrs);
  } else {
    d = da + db;
  }
    
  return -d;
}

void
apply_inverse_rotation(
    double p[3], double theta, double phi, double psi, double pt[3])
{
  double ctheta = std::cos(theta);
  double stheta = std::sin(theta);
  double cphi = std::cos(phi);
  double sphi = std::sin(phi);
  double cpsi = std::cos(psi);
  double spsi = std::sin(psi);
    
  pt[0] = ctheta*cphi*p[0] + (spsi*stheta*cphi - cpsi*sphi)*p[1] + (spsi*stheta*cphi + spsi*sphi)*p[2];
  pt[1] = ctheta*sphi*p[0] + (spsi*stheta*sphi + cpsi*cphi)*p[1] + (spsi*stheta*cphi - cpsi*sphi)*p[2];
  pt[2] = -stheta*p[0] + spsi*ctheta*p[1] + cpsi*ctheta*p[2];
}

double primitive_ellipsoid(
    double x, double y, double z, const double parameters[9])
{
  double cx = parameters[0];
  double cy = parameters[1];
  double cz = parameters[2];

  double rx = parameters[3];
  double ry = parameters[4];
  double rz = parameters[5];

  double theta = parameters[6];
  double phi = parameters[7];
  double psi = parameters[8];
  
  double xi = x - cx;
  double yi = y - cy;
  double zi = z - cz;
  double pi[3] = {xi, yi, zi};

  double pt[3];
  apply_inverse_rotation(pi, theta, phi, psi, pt);

  double val = (pt[0]/rx)*(pt[0]/rx) + (pt[1]/ry)*(pt[1]/ry) + (pt[2]/rz)*(pt[2]/rz) - 1.0;

  return val;
}
