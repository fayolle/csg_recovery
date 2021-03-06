#include <string>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <cmath>
#include <fstream>


using std::vector;


// The eval function is going to be generated by a script
extern double eval(double x, double y, double z);


#if 0
// For test.
// Produce a sphere.
double eval(double x, double y, double z) {
  return 1.0 - sqrt(x*x + y*y + z*z);
}
#endif


struct Options {
  Options() : xmin(-10),xmax(10),ymin(-10),ymax(10),zmin(-10),zmax(10),
              nx(64),ny(64),nz(64) {}

  Options(double xmin, double ymin, double zmin, double xmax, double ymax,
          double zmax, unsigned int nx, unsigned int ny, unsigned int nz) :
      xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax), 
      nx(nx), ny(ny), nz(nz) {}

  double xmin, xmax;
  double ymin, ymax;
  double zmin, zmax;
    
  unsigned int nx;
  unsigned int ny;
  unsigned int nz;
};


struct Point3 {
  double x, y, z;
  Point3() : x(0), y(0), z(0) {}
  Point3(double x, double y, double z) : x(x), y(y), z(z) {}
};


struct PointGrid {
  vector< vector < vector <Point3> > > grid;
  unsigned int nx;
  unsigned int ny;
  unsigned int nz;

  Point3& operator() (unsigned int i, unsigned int j, unsigned int k) {
    assert(i<nx);
    assert(j<ny);
    assert(k<nz);
    
    return grid[i][j][k];
  }

  const Point3& operator() (unsigned int i, unsigned int j, unsigned int k) const 
  {
    assert(i<nx);
    assert(j<ny);
    assert(k<nz);
    
    return grid[i][j][k];    
  }

  PointGrid(unsigned int nx, unsigned int ny, unsigned int nz) : 
      nx(nx), ny(ny), nz(nz) 
  {
    vector<Point3> zgrid(nz);
    vector< vector <Point3> > ygrid(ny, zgrid);
    for (unsigned int i = 0; i < nx; ++i) {
      grid.push_back(ygrid);
    }
  }

  PointGrid() : nx(0), ny(0), nz(0) {}

  void reserve(unsigned int nx, unsigned int ny, unsigned int nz) {
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    
    vector<Point3> zgrid(nz);
    vector< vector <Point3> > ygrid(ny, zgrid);
    for (unsigned int i = 0; i < nx; ++i) {
      grid.push_back(ygrid);
    }    
  }
};


struct ValueGrid {
  vector< vector < vector <double> > > grid;
  unsigned int nx;
  unsigned int ny;
  unsigned int nz;

  double& operator() (unsigned int i, unsigned int j, unsigned int k) {
    assert(i<nx);
    assert(j<ny);
    assert(k<nz);
    
    return grid[i][j][k];
  }

  const double& operator() (unsigned int i, unsigned int j, unsigned int k) const 
  {
    assert(i<nx);
    assert(j<ny);
    assert(k<nz);
    
    return grid[i][j][k];    
  }

  ValueGrid(unsigned int nx, unsigned int ny, unsigned int nz) : 
      nx(nx), ny(ny), nz(nz) 
  {
    vector<double> zgrid(nz);
    vector< vector <double> > ygrid(ny, zgrid);
    for (unsigned int i = 0; i < nx; ++i) {
      grid.push_back(ygrid);
    }
  }

  ValueGrid() : nx(0), ny(0), nz(0) {}

  void reserve(unsigned int nx, unsigned int ny, unsigned int nz) {
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    
    vector<double> zgrid(nz);
    vector< vector <double> > ygrid(ny, zgrid);
    for (unsigned int i = 0; i < nx; ++i) {
      grid.push_back(ygrid);
    }    
  }
};


void sample(const PointGrid& points, ValueGrid& samples) {
  // Assume that the function to be sampled is called eval().
  // [The function eval() is going to be generated by a script]

  assert(points.nx == samples.nx);
  assert(points.ny == samples.ny);
  assert(points.nz == samples.nz);
  
  unsigned int nx = points.nx;
  unsigned int ny = points.ny;
  unsigned int nz = points.nz;

  for (unsigned int i = 0; i < nx; ++i) {
    for (unsigned int j = 0; j < ny; ++j) {
      for (unsigned int k = 0; k < nz; ++k) {
        Point3 p = points(i,j,k);
        samples(i,j,k) = eval(p.x, p.y, p.z);
      }
    }
  }
}


void create_PointGrid(const Options& opt, PointGrid& grid) {
  unsigned int nx = opt.nx;
  double xmin = opt.xmin;
  double xmax = opt.xmax;
  // there are nx points including xmin and xmax; this means nx - 2 points in
  // the middle. This gives nx - 1 intervals. The same goes for ny and nz.
  unsigned int num_intervals_x = nx - 1;
  // length of one interval:
  double deltax = (xmax - xmin) / num_intervals_x;
  
  unsigned int ny = opt.ny;
  double ymin = opt.ymin;
  double ymax = opt.ymax;
  unsigned int num_intervals_y = ny - 1;
  double deltay = (ymax - ymin) / num_intervals_y;

  unsigned int nz = opt.nz;
  double zmin = opt.zmin;
  double zmax = opt.zmax;
  unsigned int num_intervals_z = nz - 1;
  double deltaz = (zmax - zmin) / num_intervals_z;
  

  grid.reserve(nx, ny, nz);

  
  double x;
  double y;
  double z;
  
  for (unsigned int ix = 0; ix < nx; ++ix) {
    for (unsigned int jy = 0; jy < ny; ++jy) {
      for (unsigned int kz = 0; kz < nz; ++kz) {
        x = xmin + ix * deltax;
        y = ymin + jy * deltay;
        z = zmin + kz * deltaz;
        Point3 p(x,y,z);
        grid(ix,jy,kz) = p;
      }
    }
  }
}


void create_ValueGrid(const Options& opt, ValueGrid& grid) {
  unsigned int nx = opt.nx;
  unsigned int ny = opt.ny;
  unsigned int nz = opt.nz;

  grid.reserve(nx, ny, nz);
  
  for (unsigned int ix = 0; ix < nx; ++ix) {
    for (unsigned int jy = 0; jy < ny; ++jy) {
      for (unsigned int kz = 0; kz < nz; ++kz) {
        grid(ix,jy,kz) = 0.0;
      }
    }
  }
}


// TODO
// Currently it does not work. I don't understand how this .vtr file format
// works (and I could not find a documentation either).
void
save_array_to_vtk_xml(const ValueGrid& data, const PointGrid& points, 
                      const std::string& vtk_filename) {
  unsigned int nx = data.nx;
  unsigned int ny = data.ny;
  unsigned int nz = data.nz;
  int vtk_extents[] = {0, nx-1, 0, ny-1, 0, nz-1};

  std::ofstream out(vtk_filename.c_str());
  assert(out);
  
  out << "<?xml version=\"1.0\"?>" << std::endl;
  out << "<VTKFile type=\"RectilinearGrid\">" << std::endl;
  out << "<RectilinearGrid WholeExtent=\"0 "
      << nx - 1
      << " 0 " << ny - 1
      << " 0 " << nz - 1
      << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";

  out << "<Piece Extent=\"0 "
      << nx - 1
      << " 0 " << ny - 1
      << " 0 " << nz - 1
      << "\">\n";
  
  out << "<PointData Scalars=\"Distance\">" << std::endl; //Needed?

  out << "<DataArray type=\"Float64\" Name=\"Distance\" format=\"ascii\">\n";

  for (unsigned int i = 0; i < nx; ++i) {
    for (unsigned int j = 0; j < ny; ++j) {
      for (unsigned int k = 0; k < nz; ++k) {
        out << data(i,j,k) << std::endl;
      }
    }
  }
  
  out << "</DataArray>" << std::endl;
  out << "</PointData>" << std::endl;

  out << "<Coordinates>" << std::endl;

  // X coordinates
  out << "<DataArray type=\"Float64\">" << "\n";
  for (unsigned int i = 0; i < nx; ++i) {
    for (unsigned int j = 0; j < ny; ++j) {
      for (unsigned int k = 0; k < nz; ++k) {
        Point3 p = points(i,j,k);
        out << p.x << std::endl;
      }
    }
  }
  out << "</DataArray>" << "\n";

  // Y coordinates
  out << "<DataArray type=\"Float64\">" << "\n";
  for (unsigned int i = 0; i < nx; ++i) {
    for (unsigned int j = 0; j < ny; ++j) {
      for (unsigned int k = 0; k < nz; ++k) {
        Point3 p = points(i,j,k);
        out << p.y << std::endl;
      }
    }
  }
  out << "</DataArray>" << "\n";
  
  // Z coordinates
  out << "<DataArray type=\"Float64\">" << "\n";
  for (unsigned int i = 0; i < nx; ++i) {
    for (unsigned int j = 0; j < ny; ++j) {
      for (unsigned int k = 0; k < nz; ++k) {
        Point3 p = points(i,j,k);
        out << p.z << std::endl;
      }
    }
  }
  out << "</DataArray>" << "\n";

  
  out << "</Coordinates>" << std::endl;
  
  out << "</Piece>" << std::endl;
  out << "</RectilinearGrid>\n";
  out << "</VTKFile>\n";
  
  out.close();
}


void
save_array_to_vtk(const ValueGrid& data, const PointGrid& points, 
                  const std::string& vtk_filename)
{
  std::ofstream out(vtk_filename.c_str());
  assert(out);

  unsigned int nx = data.nx;
  unsigned int ny = data.ny;
  unsigned int nz = data.nz;
  
  // header
  out << "# vtk DataFile Version 3.0" << std::endl;
  out << "vtk output" << std::endl;
  out << "ASCII" << std::endl;
  out << "DATASET STRUCTURED_GRID" << std::endl;
  out << "DIMENSIONS " << nx << " " << ny << " " << nz << std::endl;
  out << "POINTS " << nx*ny*nz << " double" << std::endl;

  // structured grid
  for (unsigned int i = 0; i < nx; ++i) {
    for (unsigned int j = 0; j < ny; ++j) {
      for (unsigned int k = 0; k < nz; ++k) {
        Point3 p = points(i, j, k);
        out << p.x << " " << p.y << " " << p.z << std::endl;
      }
    }
  }

  // data
  // header
  out << std::endl;
  out << "POINT_DATA " << nx*ny*nz << std::endl;
  out << "SCALARS Density double" << std::endl;
  out << "LOOKUP_TABLE default" << std::endl;

  // data
  for (unsigned int i =0; i < nx; ++i) {
    for(unsigned int j = 0; j < ny; ++j) {
      for (unsigned int k = 0; k < nz; ++k) {
        out << data(i,j,k) << std::endl;
      }
    }
  }

  out << std::endl;
    
  out.close();
}


void usage(const std::string& progname) {
  std::cout << "Usage:" << std::endl;
  std::cout << progname << " xmin ymin zmin xmax ymax zmax "
            << "nx ny nz output.vtk"
            << std::endl;
}


int main(int argc, char** argv) {
  // read options
  int num_args = argc - 1;
  if (num_args != 10) {
    usage(argv[0]);
    return 1;
  }
    
  // progname xmin ymin zmin xmax ymax zmax nx ny nz output.vtk
  double xmin = atof(argv[1]);
  double ymin = atof(argv[2]);
  double zmin = atof(argv[3]);
  double xmax = atof(argv[4]);
  double ymax = atof(argv[5]);
  double zmax = atof(argv[6]);
    
  unsigned int nx = static_cast<unsigned int>(atoi(argv[7]));
  unsigned int ny = static_cast<unsigned int>(atoi(argv[8]));
  unsigned int nz = static_cast<unsigned int>(atoi(argv[9]));
    
  std::string output = argv[10];
    
  Options opt(xmin, ymin, zmin, xmax, ymax, zmax, nx, ny, nz);
    
    
  // create grid of evaluation points and grid of samples from the options
  PointGrid p;
  create_PointGrid(opt, p);

  ValueGrid v;
  create_ValueGrid(opt, v);
  
  // sample eval() on the grid
  sample(p, v);
  
  // save the result in vtk format
  save_array_to_vtk(v, p, output);

  return 0;
}

