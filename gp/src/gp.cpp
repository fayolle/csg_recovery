#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <utility> // for std::pair
#include <limits> // for numeric_limits

#include <ctime>

// This code requires boost for smart pointer
#include <boost/shared_ptr.hpp>

// For random number generator
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int_distribution.hpp>


#include "stats.h"


// Control the amount of information printed.
// 0: nothing
// >0: print raw scores stats
int g_info_level = 0;


// Some utilities

template <typename T1, typename T2, typename T3>
struct triple {
  triple() {}
  
  triple(const T1& f, const T2& s, const T3& t)
      : first(f), second(s), third(t) {}
  
  T1 first;
  T2 second;
  T3 third;
};

template <typename T1, typename T2, typename T3>
triple<T1, T2, T3>
make_triple(const T1& t1, const T2& t2, const T3& t3) {
  return triple<T1, T2, T3>(t1, t2, t3);
}


// Geometric data-structures

struct Point3 {
  Point3() : _x(0.0), _y(0.0), _z(0.0) {}
  Point3(float x, float y, float z) : _x(x), _y(y), _z(z) {}
  
  float _x;
  float _y;
  float _z;
};

Point3 operator+ (const Point3& a, const Point3& b)
{
  return Point3(a._x + b._x, a._y + b._y, a._z + b._z);
}

std::vector<Point3>
operator+ (const std::vector<Point3>& a, const Point3& b)
{
  unsigned int n = a.size();
  std::vector<Point3> r(n);
  for (unsigned int i = 0; i < n; ++i) {
    r[i] = a[i] + b;
  }
  return r;
}

struct Vec3 {
  Vec3() : _x(0.0), _y(0.0), _z(0.0) {}
  Vec3(float x, float y, float z) : _x(x), _y(y), _z(z) {}
    
  float _x;
  float _y;
  float _z;
};


// Note: it is maybe better to have a vector of pair (Point3, Vec3)?
struct PointCloud {
  std::vector< Point3 > _points;
  std::vector< Vec3 > _normals;
};


std::vector<float>
operator/ (const std::vector<float>& lhs, float rhs)
{
  unsigned int n = lhs.size();
  std::vector<float> r(n);
  for (unsigned int i = 0; i < n; ++i) {
    r[i] = lhs[i] / rhs;
  }
  return r;
}

std::vector<float>
operator* (const std::vector<float>& lhs, const std::vector<float>& rhs)
{
  assert(lhs.size() == rhs.size());
  unsigned int n = lhs.size();
  std::vector<float> r(n);
  for (unsigned int i = 0; i < n; ++i) {
    r[i] = lhs[i] * rhs[i];
  }
  return r;
}

std::vector<float>
operator- (const std::vector<float>& lhs, const std::vector<float>& rhs)
{
  assert(lhs.size() == rhs.size());
  unsigned int n = lhs.size();
  std::vector<float> r(n);
  for (unsigned int i = 0; i < n; ++i) {
    r[i] = lhs[i] - rhs[i];
  }
  return r;
}

std::vector<float>
exp(const std::vector<float>& arg)
{
  unsigned int n = arg.size();
  std::vector<float> r(n);
  for (unsigned int i = 0; i < n; ++i) {
    r[i] = expf(arg[i]);
  }
  return r;
}

float sum(const std::vector<float>& arg)
{
  float s = 0.0f;
  for (std::vector<float>::const_iterator cit = arg.begin();
       cit != arg.end();
       ++cit)
  {
    s += *cit;
  }
  return s;
}

std::vector<float>
acos(const std::vector<float>& arg)
{
  unsigned int n = arg.size();
  std::vector<float> r(n);
  for (unsigned int i = 0; i < n; ++i) {
    r[i] = acosf(arg[i]);
  }
  return r;
}


Vec3 normalize(const Vec3& v);

bool read_xyzn(const std::string& filename, PointCloud& pc)
{
  std::ifstream ifile(filename.c_str());
  if (!ifile) {
    std::cerr << "Cannot open file " << filename << std::endl;
    return false;
  }

  std::string line;
  std::vector< Point3 > points;
  std::vector< Vec3 > normals;

  while (std::getline(ifile, line)) {
    float x, y, z;
    float nx, ny, nz;
    if (std::istringstream(line) >> x >> y >> z >> nx >> ny >> nz) {
      pc._points.push_back(Point3(x,y,z));
      Vec3 normal(nx, ny, nz);
      // pc._normals.push_back(Vec3(nx,ny,nz));
      pc._normals.push_back(normalize(normal));
    } else {
      std::cerr << "Error while reading xyzn file" << std::endl;
      ifile.close();
      return false;
    }
  }

  ifile.close();
  return true;
}


class Primitive {
 public:
  virtual float distance(const Point3& p) = 0;
  virtual std::vector<float> distance(const std::vector<Point3>& p)
  {
    unsigned int n = p.size();
    std::vector<float> result(n);
    for (unsigned int i = 0; i < n; ++i) {
      result[i] = distance(p[i]);
    }
    return result;
  }

  virtual float signed_distance(const Point3& p) = 0;
  virtual std::vector<float> signed_distance(const std::vector<Point3>& p)
  {
    unsigned int n = p.size();
    std::vector<float> result(n);
    for (unsigned int i = 0; i < n; ++i) {
      result[i] = signed_distance(p[i]);
    }

    return result;
  }

  virtual Vec3 gradient(const Point3& p) = 0;
  virtual std::vector<Vec3> gradient(const std::vector<Point3>& p)
  {
    unsigned int n = p.size();
    std::vector<Vec3> result(n);
    for (unsigned int i = 0; i < n; ++i) {
      result[i] = gradient(p[i]);
    }
    return result;
  }

  virtual std::string identifier() = 0;
  
};


float dot_product(float u[3], const Point3& v) {
  return u[0]*v._x + u[1]*v._y + u[2]*v._z;
}

// defined below
float dot_product(const Vec3& n, const Vec3& p);

std::vector<float>
dot_product(const std::vector<Vec3>& u, const std::vector<Vec3>& v)
{
  unsigned int n = u.size();
  std::vector<float> result(n);
  for (unsigned int i = 0; i < n; ++i) {
    result[i] = dot_product(u[i], v[i]);
  }
  return result;
}

float dot_product(const Vec3& n, const Point3& p) {
  return n._x * p._x + n._y * p._y + n._z * p._z;
}

class PlanePrimitive: public Primitive {
 public:
  PlanePrimitive(const std::vector<float>& parameters) {
    _normal_vec._x = parameters[0];
    _normal_vec._y = parameters[1];
    _normal_vec._z = parameters[2];
    _dist = parameters[3];
  }

  virtual float distance(const Point3& p) {
    return fabsf(_dist - dot_product(_normal_vec, p));
  }

  virtual float signed_distance(const Point3& p) {
    float d = dot_product(_normal_vec, p) - _dist;
    return -d;
  }

  virtual Vec3 gradient(const Point3& p) {
    return _normal_vec;
  }

  virtual std::string identifier() {
    return "plane";
  }
  
 private:
  Vec3 _normal_vec;
  float _dist;
};

Vec3 operator-(const Point3& u, const Point3& v) {
  return Vec3(u._x-v._x, u._y-v._y, u._z-v._z);
}

float norm2(const Vec3& v){
  return sqrtf(v._x*v._x + v._y*v._y + v._z*v._z);
}

Vec3 operator/(const Vec3& v, float n) {
  return Vec3(v._x / n, v._y / n, v._z / n);
}

Vec3 normalize(const Vec3& v) {
  float n = norm2(v);
  if (fabsf(n) > 1e-7)
    return v / n;
  return v;
}

std::vector<Vec3> normalize(const std::vector<Vec3>& v)
{
  unsigned int n = v.size();
  std::vector<Vec3> results(n);
  for (unsigned int i = 0; i < n; ++i) {
    results[i] = normalize(v[i]);
  }
  return results;
}

class SpherePrimitive: public Primitive {
 public:
  SpherePrimitive(const std::vector<float>& parameters) {
    _center._x = parameters[0];
    _center._y = parameters[1];
    _center._z = parameters[2];
    _radius = parameters[3];
  }

  virtual float distance(const Point3& p) {
    return fabsf(norm2(_center - p) - _radius);
  }

  virtual float signed_distance(const Point3& p) {
    Vec3 v = _center - p;
    float d = norm2(v) - _radius;
    return -d;
  }

  virtual Vec3 gradient(const Point3& p) {
    Vec3 n = _center - p;
    return normalize(n);
  }

  virtual std::string identifier() {
    return "sphere";
  }
  
 private:
  Point3 _center;
  float _radius;
};

float dot_product(const Vec3& u, const Vec3& v) {
  return u._x * v._x + u._y * v._y + u._z * v._z;
}

Vec3 operator- (const Vec3& u, const Vec3& v) {
  return Vec3(u._x - v._x, u._y - v._y, u._z - v._z);
}

Vec3 operator* (float s, const Vec3& u) {
  return Vec3(s * u._x, s * u._y, s * u._z);
}

class CylinderPrimitive: public Primitive {
 public:
  CylinderPrimitive(const std::vector<float>& parameters) {
    _axis_dir._x = parameters[0];
    _axis_dir._y = parameters[1];
    _axis_dir._z = parameters[2];

    _axis_pos._x = parameters[3];
    _axis_pos._y = parameters[4];
    _axis_pos._z = parameters[5];

    _radius = parameters[6];
  }

  virtual float distance(const Point3& p) {
    Vec3 diff = p - _axis_pos;
    float lambda = dot_product(_axis_dir, diff);
    Vec3 v = diff - lambda * _axis_dir;
    float axis_dist = norm2(v);
    return fabsf(axis_dist - _radius);
  }

  virtual float signed_distance(const Point3& p) {
    Vec3 diff = p - _axis_pos;
    float lambda = dot_product(_axis_dir, diff);
    Vec3 v = diff - lambda * _axis_dir;
    float axis_dist = norm2(v);
    float d = axis_dist - _radius;
    return -d;
  }

  virtual Vec3 gradient(const Point3& p) {
    Vec3 diff = p - _axis_pos;
    float lambda = dot_product(_axis_dir, diff);
    Vec3 normal_vec = diff - lambda * _axis_dir;
    return normalize(normal_vec);
  }

  virtual std::string identifier() {
    return "cylinder";
  }
  
 private:
  Vec3 _axis_dir;
  Point3 _axis_pos;
  float _radius;
};


Vec3 operator+ (const Vec3& u, const Vec3& v) {
  return Vec3(u._x + v._x, u._y + v._y, u._z + v._z);
}


Vec3 cross_product(const Vec3& u, const Vec3& v) {
  float ux = u._x;
  float uy = u._y;
  float uz = u._z;
  float vx = v._x;
  float vy = v._y;
  float vz = v._z;

  return Vec3(-(uz * vy) + uy * vz, uz * vx - ux * vz, -(uy * vx) + ux * vy);
}


class TorusPrimitive: public Primitive {
 public:
  TorusPrimitive(const std::vector<float>& parameters)
      : _normal_vec(parameters[0], parameters[1], parameters[2]),
        _center(parameters[3], parameters[4], parameters[5]),
        _rminor(parameters[6]), _rmajor(parameters[7])
  {}

  virtual float distance(const Point3& p) {
    Vec3 s = p - _center;
    float spin1 = dot_product(_normal_vec, s);
    Vec3 spin0vec = s - spin1*_normal_vec;
    float spin0 = norm2(spin0vec);
    spin0 = spin0 - _rmajor;
    return fabsf(sqrtf(spin0*spin0 + spin1*spin1) - _rminor);
  }

  virtual float signed_distance(const Point3& p) {
    Vec3 s = p - _center;
    float spin1 = dot_product(_normal_vec, s);
    Vec3 spin0vec = s - spin1*_normal_vec;
    float spin0 = norm2(spin0vec);
    spin0 = spin0 - _rmajor;
    float d = sqrtf(spin0*spin0 + spin1*spin1) - _rminor;
    return -d;
  }

  virtual Vec3 gradient(const Point3& p) {
    Vec3 s = p - _center;
    float spin1 = dot_product(_normal_vec, s);
    Vec3 tmp = spin1 * _normal_vec;
    Vec3 spin0vec = s - tmp;
    float spin0 = norm2(spin0vec);
    spin0 = spin0 - _rmajor;
    Vec3 pln = cross_product(s, _normal_vec);
    Vec3 plx = cross_product(_normal_vec, pln);
    plx = normalize(plx);
    Vec3 n = spin0 * plx + tmp;
    n = 1.0/sqrtf(spin0*spin0 + spin1*spin1) * n;
    return n;
  }

  virtual std::string identifier() {
    return "torus";
  }

                                
 private:
  Vec3 _normal_vec;
  Point3 _center;
  float _rminor;
  float _rmajor;
};


class ConePrimitive: public Primitive {
 public:
  ConePrimitive(const std::vector<float>& parameters)
      : _axis_dir(parameters[0], parameters[1], parameters[2]),
        _center(parameters[3], parameters[4], parameters[5]),
        _angle(parameters[6])
  {}

  virtual float distance(const Point3& p) {
    Vec3 s = p - _center;
    float g = dot_product(s, _axis_dir);
    float slen = norm2(s);
    float sqrs = slen*slen;
    float f = sqrs - g*g;
    if (f <= 0.0) {
      f = 0.0;
    } else {
      f = sqrt(f);
    }

    float da = cos(_angle) * f;
    float db = -sin(_angle) * g;

    if (g < 0.0 && (da - db) < 0.0) {
      return sqrtf(sqrs);
    }

    return fabsf(da + db);
  }

  virtual float signed_distance(const Point3& p) {
    Vec3 s = p - _center;
    float g = dot_product(s, _axis_dir);
    float slen = norm2(s);
    float sqrs = slen*slen;
    float f = sqrs - g*g;
    if (f <= 0.0) {
      f = 0.0;
    } else {
      f = sqrt(f);
    }

    float da = cos(_angle) * f;
    float db = -sin(_angle) * g;

    if (g < 0.0 && (da - db) < 0.0) {
      return -sqrtf(sqrs);
    }

    return -(da + db);
  }

  virtual Vec3 gradient(const Point3& p) {
    Vec3 s = p - _center;
    Vec3 pln = cross_product(s, _axis_dir);
    Vec3 plx = cross_product(_axis_dir, pln);
    plx = normalize(plx);
    float n0 = cos(-_angle);
    float sa = sin(-_angle);
    Vec3 ny = sa*_axis_dir;
    Vec3 n = n0*plx + ny;
    return n;
  }

  virtual std::string identifier() {
    return "cone";
  }

 private:
  Vec3 _axis_dir;
  Point3 _center;
  float _angle;
};


float bounding_box_diag_len(const PointCloud& pc) {
    
  // Get the bounding box
  float max_flt = std::numeric_limits<float>::max();
  Point3 cu(-max_flt, -max_flt, -max_flt);
  Point3 cl(max_flt, max_flt, max_flt);
    
  for (unsigned int i = 0; i < pc._points.size(); ++i) {
    Point3 p = pc._points[i];
    cu._x = std::max(cu._x, p._x);
    cu._y = std::max(cu._y, p._y);
    cu._z = std::max(cu._z, p._z);
        
    cl._x = std::min(cl._x, p._x);
    cl._y = std::min(cl._y, p._y);
    cl._z = std::min(cl._z, p._z);
  }
    
  // Get the length
  Vec3 diag = cu - cl;
  return norm2(diag);
}



std::vector<float>
min(const std::vector<float>& f0, const std::vector<float>& f1)
{
  assert(f0.size() == f1.size());
  unsigned int n = f0.size();
  std::vector<float> result(n);
  for (unsigned int i = 0; i < n; ++i) {
    result[i] = std::min(f0[i], f1[i]);
  }
  return result;
}

std::vector<float>
min(const std::vector<float>& f0, float f1)
{
  unsigned int n = f0.size();
  std::vector<float> result(n);
  for (unsigned int i = 0; i < n; ++i) {
    result[i] = std::min(f0[i], f1);
  }
  return result;
}

std::vector<float>
max(const std::vector<float>& f0, const std::vector<float>& f1)
{
  assert(f0.size() == f1.size());
  unsigned int n = f0.size();
  std::vector<float> result(n);
  for (unsigned int i = 0; i < n; ++i) {
    result[i] = std::max(f0[i], f1[i]);
  }
  return result;
}

std::vector<float>
max(const std::vector<float>& f0, float f1)
{
  unsigned int n = f0.size();
  std::vector<float> result(n);
  for (unsigned int i = 0; i < n; ++i) {
    result[i] = std::max(f0[i], f1);
  }
  return result;
}

std::vector<float> operator-(const std::vector<float>& f)
{
  unsigned int n = f.size();
  std::vector<float> result(n);
  for (unsigned int i = 0; i < n; ++i) {
    result[i] = -f[i];
  }
  return result;
}


class Operation {
 public:
  virtual float evaluate(const std::vector<float>& results) = 0;
  // vectorized version
  virtual std::vector<float>
  evaluate(const std::vector< std::vector<float> >& results) = 0;

  virtual int get_childcount() = 0;
  virtual std::string get_name() = 0;
};


class UnionOp: public Operation {
 public:
  virtual float evaluate(const std::vector<float>& results) {
    float f0 = results[0];
    float f1 = results[1];
    //return f0 + f1 + sqrtf(f0*f0 + f1*f1);
    return std::max(f0, f1);
  }

  virtual std::vector<float>
  evaluate(const std::vector< std::vector<float> >& results)
  {
    std::vector<float> f0 = results[0];
    std::vector<float> f1 = results[1];
    return max(f0, f1);
  }
  
  virtual int get_childcount() { return 2; }

  virtual std::string get_name() { return "union"; }
};


class IntersectionOp: public Operation {
 public:
  virtual float evaluate(const std::vector<float>& results) {
    float f0 = results[0];
    float f1 = results[1];
    //return f0 + f1 - sqrtf(f0*f0 + f1*f1);
    return std::min(f0, f1);
  }

  virtual std::vector<float>
  evaluate(const std::vector< std::vector<float> >& results)
  {
    std::vector<float> f0 = results[0];
    std::vector<float> f1 = results[1];
    return min(f0, f1);
  }
  
  virtual int get_childcount() { return 2; }

  virtual std::string get_name() { return "intersection"; }
};


class SubtractionOp: public Operation {
 public:
  virtual float evaluate(const std::vector<float>& results) {
    float f0 = results[0];
    float f1 = results[1];
    //return f0 - f1 - sqrtf(f0*f0 + f1*f1);
    return std::min(f0, -f1);
  }

  virtual std::vector<float>
  evaluate(const std::vector< std::vector<float> >& results)
  {
    std::vector<float> f0 = results[0];
    std::vector<float> f1 = results[1];
    return min(f0, -f1);
  }
  
  virtual int get_childcount() { return 2; }
  
  virtual std::string get_name() { return "subtraction"; }
};


class NegationOp: public Operation {
 public:
  virtual float evaluate(const std::vector<float>& results) {
    return -results[0];
  }

  virtual std::vector<float>
  evaluate(const std::vector< std::vector<float> >& results)
  {
    std::vector<float> f0 = results[0];
    return -f0;
  }
  
  virtual int get_childcount() { return 1; }
  
  virtual std::string get_name() { return "negation"; }
};


typedef boost::shared_ptr<Operation> OperationPtr;

class Node;
typedef boost::shared_ptr<Node> NodePtr;
typedef NodePtr Tree;

class LeafNode;
typedef boost::shared_ptr<LeafNode> LeafNodePtr;
class InternalNode;
typedef boost::shared_ptr<InternalNode> InternalNodePtr;


typedef boost::shared_ptr<Primitive> PrimitivePtr;


class Node {
 public:
  virtual float evaluate(const Point3& p) = 0;
  virtual std::vector<float> evaluate(const std::vector<Point3>& p) = 0;
  virtual int compute_number_nodes() = 0;
  virtual int max_depth() = 0;
  virtual std::string to_string() = 0;
  virtual void add_child(NodePtr n) = 0;
  virtual NodePtr copy() = 0;
  virtual NodePtr find_node_at(int point) = 0;
  virtual NodePtr replace(NodePtr new_node, int point) = 0;
  virtual NodePtr replace_rec(NodePtr new_node, int point) = 0;
};


class InternalNode: public Node {
 public:
  InternalNode(const OperationPtr& op) {
    _operation = op;
    _name = op->get_name();
  }
  
  // Deep copy of the tree
  virtual NodePtr copy() {
    NodePtr n(new InternalNode(_operation));
    for (unsigned int i = 0; i < _children.size(); ++i) {
      n->add_child(_children[i]->copy());
    }
    return n;
  }
  

  virtual void add_child(NodePtr n) {
    _children.push_back(n);
  }

  virtual float evaluate(const Point3& p) {
    assert(_children.size() == 
           static_cast<unsigned int>(_operation->get_childcount()));

    std::vector<float> results;
    for (unsigned int i = 0; i < _children.size(); ++i) {
      results.push_back(_children[i]->evaluate(p));
    }
    return _operation->evaluate(results);
  }

  virtual std::vector<float> evaluate(const std::vector<Point3>& p)
  {
    assert(_children.size() ==
           static_cast<unsigned int>(_operation->get_childcount()));

    std::vector< std::vector<float> > results;
    for (unsigned int i = 0; i < _children.size(); ++i) {
      results.push_back(_children[i]->evaluate(p));
    }

    return _operation->evaluate(results);
  }
  
  virtual std::string to_string() {
    std::string str_to_display = _name + "[";
    for (unsigned int i = 0; i < _children.size() - 1; ++i) {
      str_to_display = str_to_display + _children[i]->to_string() + ",";
    }
    str_to_display = str_to_display + _children[_children.size()-1]->to_string();
    str_to_display = str_to_display + "]";
    
    return str_to_display;
  }
  
  virtual int compute_number_nodes() {
    int number_nodes = 0;
    for (unsigned int i = 0; i < _children.size(); ++i) {
      number_nodes = number_nodes + _children[i]->compute_number_nodes();
    }
    return 1 + number_nodes;
  }

  virtual int max_depth() {
    int max_depth_children = 0;
    for (unsigned int i = 0; i < _children.size(); ++i) {
      max_depth_children = std::max(max_depth_children, _children[i]->max_depth());
    }
    return 1+max_depth_children;
  }

  virtual NodePtr find_node_at(int point)
  {
    if (point == 0) {
      return this->copy();
    } else {
      for (unsigned int i = 0; i < _children.size(); ++i) {
        int number_nodes = _children[i]->compute_number_nodes();
        if (point <= number_nodes) {
          return _children[i]->find_node_at(point-1);
        } else {
          point = point - number_nodes;
        }
      }
    }
  }

  virtual NodePtr replace(NodePtr new_node, int point)
  {
    Tree tree_copy = this->copy();
    return tree_copy->replace_rec(new_node, point);
  }
  
  virtual NodePtr replace_rec(NodePtr new_node, int point)
  {
    if (point == 0) {
      return new_node;
    } else {
      for (unsigned int i = 0; i < _children.size(); ++i) {
        int number_nodes = _children[i]->compute_number_nodes();
        if (point <= number_nodes) {
          _children[i] = _children[i]->replace_rec(new_node, point-1);
          Tree this_copy = this->copy();
          return this_copy;
        } else {
          point = point - number_nodes;
        }
      }
    }
    
  }
  
 private:
  std::vector<NodePtr> _children;
  std::string _name;
  OperationPtr _operation;
};


class LeafNode: public Node {
 public:
  LeafNode(const PrimitivePtr& p, const std::string& name) {
    _primitive = p;
    _name = name;
  }

  virtual float evaluate(const Point3& p) {
    return _primitive->signed_distance(p);
  }

  virtual std::vector<float> evaluate(const std::vector<Point3>& p) {
    return _primitive->signed_distance(p);
  }
  
  virtual std::string to_string() {
    return _name;
  }

  virtual int compute_number_nodes() {
    return 1;
  }

  // return the depth of the deepest branch of the tree
  virtual int max_depth() {
    return 1;
  }
  
  virtual void add_child(NodePtr n) {
    // do nothing as a leaf has no children
  }

  virtual NodePtr copy() {
    NodePtr n(new LeafNode(_primitive, _name));
    return n;
  }

  virtual NodePtr find_node_at(int point) {
    assert(point == 0);
    return this->copy();
  }

  virtual NodePtr replace(NodePtr new_node, int point)
  {
    assert(point == 0);
    Tree tree_copy = this->copy();
    return tree_copy->replace_rec(new_node, point);
  }

  virtual NodePtr replace_rec(NodePtr new_node, int point)
  {
    assert(point == 0);
    return new_node;
  }
  
 private:
  PrimitivePtr _primitive;
  std::string _name;
};





PrimitivePtr create_primitive_instance(const std::string& name, std::vector<float>& parameters)
{
  if (name == "plane") {
    PrimitivePtr plane(new PlanePrimitive(parameters));
    return plane;
  } else if (name == "cylinder") {
    PrimitivePtr cyl(new CylinderPrimitive(parameters));
    return cyl;
  } else if (name == "sphere") {
    PrimitivePtr sphere(new SpherePrimitive(parameters));
    return sphere;
  } else if (name == "torus") {
    PrimitivePtr torus(new TorusPrimitive(parameters));
    return torus;
  } else if (name == "cone") {
    PrimitivePtr cone(new ConePrimitive(parameters));
    return cone;
  }
}


bool
read_fit(
    const std::string& filename, std::vector<PrimitivePtr>& primitive_list)
{
  std::ifstream in(filename.c_str());
  if (in.fail()) return false;

  std::string line;

  while (std::getline(in, line)) {
    std::stringstream split(line);
    std::vector<std::string> splitted_line;
    std::istream_iterator<std::string> split_it(split);
    std::istream_iterator<std::string> eof;

    std::copy(split_it, eof, std::back_inserter(splitted_line));

    //    std::cout << splitted_line.size() << std::endl;
    if (splitted_line.size() == 0) {
      // skip empty line
    } else {
      std::string name = splitted_line[0];
      assert(name == "plane" || name == "sphere" || name == "cylinder"
             || name == "cone" || name == "torus");

      std::vector<float> parameters;
      for (unsigned int i = 1; i < splitted_line.size(); ++i) {
        parameters.push_back(atof(splitted_line[i].c_str()));
      }

      primitive_list.push_back(create_primitive_instance(name, parameters));
    }
  }
  
  in.close();
  return true;
}


void
create_leaves(
    const std::vector<PrimitivePtr>& primitive_list,
    std::vector<LeafNodePtr>& leaves)
{
  int count = 0;
  for (unsigned int i = 0; i < primitive_list.size(); ++i) {
    std::stringstream tmp;
    tmp << primitive_list[i]->identifier() << count;
    std::string name = tmp.str();
    LeafNodePtr leaf(new LeafNode(primitive_list[i], name));
    leaves.push_back(leaf);
    count = count + 1;
  }
}


void create_operations(std::vector<OperationPtr>& operations) {
  OperationPtr uni(new UnionOp());
  operations.push_back(uni);
  
  OperationPtr inter(new IntersectionOp());
  operations.push_back(inter);
  
  OperationPtr sub(new SubtractionOp());
  operations.push_back(sub);
  
  OperationPtr neg(new NegationOp());
  operations.push_back(neg);
}


struct CompareCreatures {
  bool
  operator() (
      const triple<float, float, Tree>& c1, const triple<float, float, Tree>& c2)
  {
    return c1.first < c2.first;
  }
};


class RankFunction {
 public:
  RankFunction(const PointCloud& pc, float eps=0.01, float n_eps=0.6) 
      : _point_cloud(pc), _epsilon(eps), _normal_epsilon(n_eps) {}


  std::vector< triple<float, float, Tree> > 
  operator() (const std::vector<Tree>& population) 
  {
    std::vector< triple<float, float, Tree> > scores;

    //clock_t b = std::clock();
    
    // evaluate the score of each tree in the population
    for (unsigned int i = 0; i < population.size(); ++i) {
      Tree c = population[i];
      float score = score_function_vectorized(c); //score_function(c);
      scores.push_back(make_triple(score, score, c));
    }

    //clock_t e = std::clock();
    //double elapsed = double(e - b) / CLOCKS_PER_SEC;
    //std::cout << "time to compute the raw obj fun " << elapsed << std::endl;


    //b = std::clock();
    
    normalize_fitness(scores);

    //e = std::clock();
    //elapsed = double(e - b) / CLOCKS_PER_SEC;
    //std::cout << "time to normalize " << elapsed << std::endl;


    //b = std::clock();
    
    // sort the normalized scores
    std::sort(scores.begin(), scores.end(), CompareCreatures());

    //e = std::clock();
    //elapsed = double(e - b) / CLOCKS_PER_SEC;
    //std::cout << "time to sort the population " << elapsed << std::endl;

    
    // return the sorted list of triples
    // (normalized score, raw score, creature)
    return scores;
  }
  
  
 private:
  PointCloud _point_cloud;
  float _epsilon;
  float _normal_epsilon;

  float score_function_vectorized(Tree& c)
  {
    float model_length = bounding_box_diag_len(_point_cloud);
    float error_epsilon = _epsilon * model_length;
    
    std::vector<float> f = c->evaluate(_point_cloud._points);
    std::vector<float> fs = f / error_epsilon;
    float dist_error = sum(exp(-fs*fs));
    
    float normal_epsilon = _normal_epsilon; 
    float delta = 0.00001f;
    float normal_error = 0.0f;
          
    std::vector<Point3> p = _point_cloud._points;
    std::vector<Point3> pdx = p + Point3(delta, 0, 0);
      
    std::vector<float> df = c->evaluate(pdx);
    std::vector<float> dfdx = (df - f) / delta;

    std::vector<Point3> pdy = p + Point3(0, delta, 0);
    df = c->evaluate(pdy);
    std::vector<float> dfdy = (df - f) / delta;
        
    std::vector<Point3> pdz = p + Point3(0, 0, delta);
    df = c->evaluate(pdz);
    std::vector<float> dfdz = (df - f) / delta;
    
    unsigned int n = df.size();
    std::vector<Vec3> gradf(n);   
    for (unsigned int i = 0; i < n; ++i) {
      gradf[i] = Vec3(-dfdx[i], -dfdy[i], -dfdz[i]);
    }

    gradf = normalize(gradf);
    std::vector<float> dot = dot_product(gradf, _point_cloud._normals);

    // make sure that dot is in [-1, 1]
    dot = min(max(dot, -1.0f), 1.0f);
    std::vector<float> angle = acos(dot);
    std::vector<float> theta = angle / normal_epsilon;
    
    normal_error = sum(exp(-theta*theta));
     
    // penalize large trees
    int num_nodes = c->compute_number_nodes();
    int num_points = _point_cloud._points.size();
    float objective_value = 
        dist_error + normal_error - (1.0f/2.0f)*num_nodes*logf(num_points);

    return std::max(objective_value, 0.0f);
  }
 
  float score_function(Tree& c) 
  {
    float model_length = bounding_box_diag_len(_point_cloud);
    float error_epsilon = _epsilon * model_length;
    
    float dist_error = 0;
    
    for (unsigned int i = 0; i < _point_cloud._points.size(); ++i) {
      Point3 p = _point_cloud._points[i];
      float v = c->evaluate(p);
      v = v / error_epsilon;
      dist_error += expf(-v*v);
    }
    
    float normal_epsilon = _normal_epsilon; 
    float delta = 0.00001f;
    float normal_error = 0.0f;
    
    for (unsigned int i = 0; i < _point_cloud._points.size(); ++i) {
      Point3 p = _point_cloud._points[i];
      Vec3 n = _point_cloud._normals[i];
        
      float f = c->evaluate(p);
        
      Point3 pdx(p._x + delta, p._y, p._z);
      float df = c->evaluate(pdx);
      float dfdx = (df - f) / delta;
        
      Point3 pdy(p._x, p._y + delta, p._z);
      df = c->evaluate(pdy);
      float dfdy = (df - f) / delta;
        
      Point3 pdz(p._x, p._y, p._z + delta);
      df = c->evaluate(pdz);
      float dfdz = (df - f) / delta;
        
      Vec3 gradf(-dfdx, -dfdy, -dfdz);
        
      gradf = normalize(gradf);
      float dot = dot_product(gradf, n);

      // make sure that dot is in [-1, 1]
      dot = std::min(std::max(dot, -1.0f), 1.0f); 
      float angle = acosf(dot);
      float theta = angle / normal_epsilon;
        
      normal_error += expf(-theta*theta);
    }

    // penalize large trees
    int num_nodes = c->compute_number_nodes();
    int num_points = _point_cloud._points.size();
    float objective_value = 
        dist_error + normal_error - (1.0f/2.0f)*num_nodes*logf(num_points);
        
    return std::max(objective_value, 0.0f);
  }
 
  // replace each score by a normalized score (in-place)
  void normalize_fitness(std::vector< triple<float, float, Tree> >& scores) 
  {
    float total_scores = 0.0f;
    for (unsigned int i = 0; i < scores.size(); ++i) {
      float score = scores[i].first;
      score = 1.0f / (1.0f + score);
      scores[i].first = score;
      total_scores = total_scores + score;
    }
   
    for (unsigned int i = 0; i < scores.size(); ++i) {
      scores[i].first = scores[i].first / total_scores;
    }
  }
};


// Print some information about the current population
void
print_information(const std::vector< triple<float, float, Tree> >& population,
                  const int info_level=0)
{
  // A triple in the population corresponds to:
  // (normalized score, raw score, creature)
  //

  if (info_level > 0) {
  
    size_t N = population.size();
    std::vector<float> raw_scores(N);

    // Gather information about the raw scores
    for (size_t i=0; i<N; ++i) {
      raw_scores[i] = population[i].second;
    }

    std::cout << "Raw scores:" << std::endl;
    float min_raw_score = *(std::min_element(raw_scores.begin(), raw_scores.end()));
    std::cout << "Min: " << min_raw_score << std::endl;
    float max_raw_score = *(std::max_element(raw_scores.begin(), raw_scores.end()));
    std::cout << "Max: " << max_raw_score << std::endl;
    float mean_raw_score = mean(raw_scores);
    float sd_raw_score = stdev(raw_scores);
    std::cout << "Mean: " << mean_raw_score
              << " "
              << "SD: " << sd_raw_score
              << std::endl;
  }
}


class GP {
 public:
  GP(const std::vector<LeafNodePtr>& leaves, const std::vector<OperationPtr>& operations, const RankFunction& rank_function) 
      : _rank_function(rank_function)
 
  {
    _list_leaves = leaves;
    _list_operations = operations;
    //_rank_function = rank_function;
  }

  void evolve(int pop_size, int max_gen = 500, float mutation_rate = 0.1, float crossover_rate = 0.4) 
  { 
    // Create a random initial population
    std::cerr << "Create random initial population" << std::endl;
    for (int i = 0; i < pop_size; ++i) {
      _population.push_back(make_random_tree(0.7, 10)); // opr=0.7, max_depth=10
    }
   
   
    // Evolution:
    std::cerr << "Evolution" << std::endl;
    for (int i = 0; i < max_gen; ++i) {
      _scores = _rank_function(_population);

      // print some information about the population:
      print_information(_scores, g_info_level);

      // print the best creature every 10 iterations:
      if ((i != 0) && (i % 10 == 0) && (g_info_level>0)) {
        Tree best_creature = _scores[0].third;
        std::cout << best_creature->to_string() << std::endl;
      }
      
      // keep the two best creatures:
      std::vector<Tree> new_pop;
      new_pop.push_back(_scores[0].third);
      new_pop.push_back(_scores[1].third);
     
      while (new_pop.size() < static_cast<unsigned int>(pop_size))
      {
        int c1_index = tournament();
        int c2_index = tournament();
        
        Tree c1 = _scores[c1_index].third;
        Tree c2 = _scores[c2_index].third;
        
        std::pair<Tree, Tree> offspring = 
            crossover(c1, c2, crossover_rate);
        Tree new_c1 = mutate(offspring.first, mutation_rate); //TODO mutation
        Tree new_c2 = mutate(offspring.second, mutation_rate);
        
        new_pop.push_back(new_c1);
        new_pop.push_back(new_c2);
      }
     
      // update the population
      _population = new_pop;
    }
  }

  void test_create_population(int pop_size) {
    // Create a random initial population
    for (int i = 0; i < pop_size; ++i) {
      // opr=0.7, max_depth=10
      _population.push_back(make_random_tree(0.7, 10)); 
    }
    
    for (int i = 0; i < pop_size; ++i) {
      std::cout << "creature " << i << std::endl;
      std::cout << _population[i]->to_string() << std::endl;
   
      Tree copy = _population[i]->copy();
      std::cout << "copy: " << std::endl;
      std::cout << copy->to_string() << std::endl;
    }
  }

  
  void test_create_and_evaluate_population(int pop_size) {
    clock_t b = std::clock();
    
    // Create a random initial population
    for (int i = 0; i < pop_size; ++i) {
      // opr=0.7, max_depth=10
      _population.push_back(make_random_tree(0.7, 10)); 
    }

    clock_t e = std::clock();
    double elapsed = double(e - b) / CLOCKS_PER_SEC;
    std::cout << "time to create population " << elapsed << std::endl;


    b = std::clock();
    
    _scores = _rank_function(_population);

    e = std::clock();
    elapsed = double(e - b) / CLOCKS_PER_SEC;
    std::cout << "time in evaluating population " << elapsed << std::endl;

    
    // print best creature and its score
    float best_raw_score = _scores[0].second;
    std::cout << best_raw_score << std::endl;
    std::cout << (_scores[0].third)->to_string() << std::endl;

    /*
      for (int i = 0; i < pop_size; ++i) {
      std::cout << "creature " << i << std::endl;
      std::cout << _population[i]->to_string() << std::endl;
      }
    */
  }
  
  
  Tree get_best_creature() 
  {
    _scores = _rank_function(_population);
    return _scores[0].third;
  }


 private:
  std::vector<LeafNodePtr> _list_leaves;
  std::vector<OperationPtr> _list_operations;
 
  std::vector<Tree> _population;
  // a score is a triple consisting in normalized score, raw score, creature
  std::vector< triple<float, float, Tree> > _scores;
 
  RankFunction _rank_function;
 
  boost::random::mt19937 _rng;
 
 
  Tree make_random_tree(float opr = 0.7, int max_depth = 4) {
    if (uniform_01() < opr && max_depth > 0) {
      int idx = uniform_int(0, _list_operations.size() - 1);
      OperationPtr op = _list_operations[idx];
        
      NodePtr n(new InternalNode(op));
        
      for (int i = 0; i < op->get_childcount(); ++i) {
        n->add_child(make_random_tree(opr, max_depth-1));
      }
        
      return n;
        
    } else {
      int idx = uniform_int(0, _list_leaves.size() - 1);
      LeafNodePtr n = _list_leaves[idx];

      return n;    
    }
    
  }
 
  // Generate a random number in [0, 1] from a uniform distribution
  float uniform_01() {
    boost::random::uniform_01<float> unif_01;
    float x = unif_01(_rng);
    return x;
  }

  // Generate a random int in [a, b] from a uniform distribution
  int uniform_int(int a, int b) {
    boost::random::uniform_int_distribution<> unif_int_distrib(a,b);
    int x = unif_int_distrib(_rng);
    return x;
  }

  // Assume _scores is a list of triple (normalized score, raw score, tree)
  // The tournament consists in selecting two creatures randomly and
  // return the index of the one with the lowest (normalized) fitness value
  // (i.e. the best creature).
  // 
  // Note 1: it assumes that _scores is sorted
  // Note 2: tournament in general involves more than 2 creatures 
  int tournament() {
    int number_creatures = _scores.size();
    int index1 = uniform_int(0, number_creatures-1);
    int index2 = uniform_int(0, number_creatures-1);
    // check that index1 and index2 are different:
    while (index1 == index2) {
      index2 = uniform_int(0, number_creatures-1);
    }

    if (_scores[index1].first < _scores[index2].first) {
      return index1;
    } else {
      return index2;
    }

    // Should never happen
    // return 0;
  }
 
 
  std::pair<Tree, Tree>
  crossover(
      Tree& c1, Tree& c2, float crossover_rate=0.5)
  {
    float r = uniform_01();
    if (r < crossover_rate) {
      return _crossover(c1, c2);
    } else {
      Tree copy_c1 = c1->copy();
      Tree copy_c2 = c2->copy();

      return std::make_pair(copy_c1, copy_c2);
    }
  }

  std::pair<Tree, Tree>
  _crossover(Tree& c1, Tree& c2)
  {
    // Make copies of the original creatures
    Tree c1_copy = c1->copy();
    Tree c2_copy = c2->copy();

    // Find number of nodes in c1 and c2
    int number_nodes_c1 = c1->compute_number_nodes();
    int number_nodes_c2 = c2->compute_number_nodes();

    // Generate random numbers in [0, #c1 - 1]
    // and [0, #c2 - 1] <-- crossover points
    int crossover_point1 = uniform_int(0, number_nodes_c1 - 1);
    int crossover_point2 = uniform_int(0, number_nodes_c2 - 1);

    // Find node of c1 at position #c1 and node of c2 at position #c2]
    NodePtr node1 = c1_copy->find_node_at(crossover_point1);
    NodePtr node2 = c2_copy->find_node_at(crossover_point2);

    // Replace nodes: #c1 of c1 with node #c2 of c2 and
    // #c2 of c2 with #c1 of c1:
    Tree new_c1 = c1_copy->replace(node2, crossover_point1);
    Tree new_c2 = c2_copy->replace(node1, crossover_point2);

    // last argument is the maximum allowed depth; I should
    // probably store it in a parameter or a constant? 
    return validate_crossover(c1_copy, new_c1, c2_copy, new_c2, 10);
  }


  // Given old (before crossover) and new (after crossover) creatures check
  // to see whether the maximum depth was exceeded in each tree.
  // If either of the new individuals has exceeded the maximum depth
  // then the old creatures are used.
  std::pair<Tree, Tree>
  validate_crossover(
      Tree& c1, Tree& new_c1, Tree& c2, Tree& new_c2, int max_depth)
  {
    // Compute max depth of new creatures
    int new_c1_depth = new_c1->max_depth();
    int new_c2_depth = new_c2->max_depth();

    // If either of the new creatures has exceeded a user defined max depth
    // then we keep the old creatures
    if ((new_c1_depth > max_depth) || (new_c2_depth > max_depth)) {
      return std::make_pair(c1, c2);
    } else {
      return std::make_pair(new_c1, new_c2);
    }
  }
  
  Tree mutate(Tree& c, float mutation_rate=0.1) {
    float r = uniform_01();
    if (r < mutation_rate) {
      return _mutate(c, 0.1);
    } else {
      return c->copy();
    }
  }

  Tree _mutate(Tree& c, float probchange=0.1) {
    float r = uniform_01();
    if (r < probchange) {
      return make_random_tree();
    } else {
      // find number of nodes in the tree
      int number_nodes = c->compute_number_nodes();
      // generate a random number between 0 and #nodes --> mutation point
      int mutation_point = uniform_int(0, number_nodes-1);
      // generate a new random tree (with limited max depth) --> new node
      Tree new_node = make_random_tree();
      // call replace with the mutation point and new node
      return c->replace(new_node, mutation_point);
    }
  }
};


void save_creature_to_file(Tree& creature, const std::string& filename)
{
  std::ofstream out(filename.c_str());

  if (out.fail()) {
    std::cerr << "Error when opening " << filename << std::endl;
    return;
  }

  out << creature->to_string() << std::endl;
  
  out.close();
}


void
save_primitive_list_to_file(
    std::vector<LeafNodePtr>& leaves, const std::string& filename)
{
  std::ofstream out(filename.c_str());

  if (out.fail()) {
    std::cerr << "Error when opening " << filename << std::endl;
    return;
  }

  unsigned int num_prim = leaves.size();
  for (unsigned int i = 0; i < num_prim - 1; ++i) {
    out << leaves[i]->to_string() << ",";
  }

  out << leaves[num_prim-1]->to_string() << std::endl;
  
  out.close();
}


void usage(const std::string& prog_name)
{
  std::cout << "Usage: " << std::endl;
  std::cout << prog_name
            << " primitives.fit point_set.xyzn best_creature.txt"
            << " primitive_list.txt print_level num_creatures" 
            << " num_iterations mutation_rate crossover_rate" 
            << std::endl;
}


int main(int argc, char** argv)
{
  if (argc!=10) {
    usage(argv[0]);
    return -1;
  }
  
  std::string fit_filename = argv[1];

  std::vector<PrimitivePtr> primitive_list;
  read_fit(fit_filename, primitive_list);
  
  std::vector<LeafNodePtr> leaves;
  create_leaves(primitive_list, leaves);

  std::vector<OperationPtr> operations;
  create_operations(operations);
  
  std::string xyzn_filename = argv[2];
  PointCloud pc;
  read_xyzn(xyzn_filename, pc);

  g_info_level = atoi(argv[5]);
    
  Tree best_creature;
  RankFunction rf(pc);
  GP gp(leaves, operations, rf);

  int pop_size = atoi(argv[6]);
  int max_iter = atoi(argv[7]);
  float mutation_rate = atof(argv[8]);
  float crossover_rate = atof(argv[9]);  
  //evolve(int pop_size, int max_gen = 500, float mutation_rate = 0.1, float crossover_rate = 0.4) 
  gp.evolve(pop_size, max_iter, mutation_rate, crossover_rate);
  best_creature = gp.get_best_creature();
  
  std::string expression_filename = argv[3];
  save_creature_to_file(best_creature, expression_filename);
  
  std::string primitive_list_filename = argv[4];
  save_primitive_list_to_file(leaves, primitive_list_filename);

  return 0;
}
