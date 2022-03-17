#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <limits>
#include <cmath>


// Sample an expression defined in the function eval at each point of
// an input point-set passed as argument.
// The computed value approximates a distance error of the model described by
// the expression defined in eval.


// The eval function is going to be generated automatically by a script.
extern double eval(double x, double y, double z);


struct Point3 {
  double x, y, z;
  Point3() : x(0), y(0), z(0) {}
  Point3(double x, double y, double z) : x(x), y(y), z(z) {}
};


typedef std::vector<Point3> PointSet;


bool read_point_set(const std::string& input_filename, PointSet& ps) {
  std::ifstream input(input_filename.c_str());
  if (!input) {
    std::cerr << "Error while opening " << input_filename;
    return false;
  }

  std::string line;
  while(std::getline(input, line)) {
    double x, y, z;
    double nx, ny, nz;

    if (std::istringstream(line) >> x >> y >> z >> nx >> ny >> nz) {
      ps.push_back(Point3(x,y,z));
    } else {
      std::cerr << "Error while reading xyzn file" << std::endl;
      input.close();
      return false;
    }
    
  }
  
  input.close();
  return true;
}


void
sample_eval_at_point_set(const PointSet& ps, std::vector<double>& values) {
  for (PointSet::const_iterator it = ps.begin(); it != ps.end(); ++it) {
    double x = it->x;
    double y = it->y;
    double z = it->z;
    double val = eval(x, y, z);
    values.push_back(std::fabs(val));
  }
}


// Rescale all values by scale
void
rescale(std::vector<double>& values, double scale) {
  for (std::vector<double>::iterator it = values.begin();
       it != values.end();
       it++)
  {
    *it = *it * scale;
  }
}


struct BoundingBox {
  double xmin, ymin, zmin;
  double xmax, ymax, zmax;

  // Default ctor for a bounding box initializes xmin, ymin
  // and zmin to +infinity
  // and xmax, ymax and zmax to -infinity.
  BoundingBox() {
    double max_flt = std::numeric_limits<double>::max();
    xmin = max_flt;
    ymin = max_flt;
    zmin = max_flt;

    xmax = -max_flt;
    ymax = -max_flt;
    zmax = -max_flt;
  }
};


void compute_bounding_box(const PointSet& ps, BoundingBox& bb) {
  // Make sure that it is initialized properly
  BoundingBox temp;
  bb = temp;
  
  for (PointSet::const_iterator it = ps.begin(); it != ps.end(); ++it) {
    // do the tests
    if (it->x < bb.xmin) bb.xmin = it->x;
    if (it->x > bb.xmax) bb.xmax = it->x;
    
    if (it->y < bb.ymin) bb.ymin = it->y;
    if (it->y > bb.ymax) bb.ymax = it->y;

    if (it->z < bb.zmin) bb.zmin = it->z;
    if (it->z > bb.zmax) bb.zmax = it->z;
  }

  // bb contains the point-set's bounding-box
}


struct Vec3{
  double x, y, z;
  // Default ctor is the null vector
  Vec3() : x(0), y(0), z(0) {}
  Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
};


// Compute the L2 norm of an input vector
double compute_length(const Vec3& v) {
  double x2 = v.x * v.x;
  double y2 = v.y * v.y;
  double z2 = v.z * v.z;

  return std::sqrt(x2 + y2 + z2);
}


// Compute the length of the diag of the ps's boundingbox
double compute_point_set_diag_length(const PointSet& ps)
{
  BoundingBox bb;
  compute_bounding_box(ps, bb);

  double len;
  len = compute_length(
      Vec3(bb.xmax-bb.xmin, bb.ymax-bb.ymin, bb.zmax-bb.zmin));

  return len;
}


// Save coordinates and function values in a comma separated values file.
// Paraview can read csv file:
// http://paraview.org/Wiki/ParaView/Data_formats
bool
write_values_csv(
    const PointSet& ps, const std::vector<double>& v,
    const std::string& csv_filename)
{
  std::ofstream out(csv_filename.c_str());
  if (!out) {
    std::cerr << "Error while writing in " << csv_filename;
    return false;
  }

  // Header
  out << "x,y,z,error" << std::endl;
  
  // Data
  for (size_t i = 0; i < ps.size(); ++i) {
    out << ps[i].x << "," << ps[i].y << "," << ps[i].z << ","
        << v[i] << std::endl;
  }
  
  out.close();
  return true;
}


void usage(const std::string& progname) {
  std::cout << "Usage: " << std::endl;
  std::cout << progname << " point_set.xyzn error.csv" << std::endl;
}


int main(int argc, char** argv) {
  int num_args = argc - 1;
  if (num_args != 2) {
    usage(argv[0]);
    return 1;
  }

  // input file name: we will apply eval() to each point in this file
  std::string xyzn_file = argv[1];

  // output file name: it will contain the coordinates of each point and the
  // value of eval() at this point
  std::string csv_file = argv[2];

  PointSet ps;
  read_point_set(xyzn_file, ps);

  std::vector<double> values;
  sample_eval_at_point_set(ps, values);

  // Rescale by the length of the diagonal of the bounding box of the object
  double diag_len = compute_point_set_diag_length(ps);
  if (diag_len == 0.0) {
    std::cerr << "Diagonal length of the point set has zero length";
    return 1;
  }
  std::cout << diag_len << std::endl;
  rescale(values, 1.0/diag_len);
  
  write_values_csv(ps, values, csv_file);

  return 0;
}
