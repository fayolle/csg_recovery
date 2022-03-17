#ifndef OBB_VEC3_H
#define OBB_VEC3_H

#include <cmath>
#include <algorithm>


// Vec3 is a common name and could be used in code where this class
// will be used as well, so I renamed it to OBBVec3.
//
class OBBVec3 {
public:
  double x;
  double y;
  double z;

  OBBVec3(double X=0.0f, double Y=0.0f, double Z=0.0f){
    Set(X,Y,Z);
  }

  void Set(double X, double Y, double Z){
    x=X; y=Y; z=Z;
  }

  double &operator[](int id) { 
    return *(&x+id); 
  }
  
  OBBVec3 operator-() { 
    return OBBVec3( -x, -y, -z ); 
  }

  OBBVec3 operator+(OBBVec3 in) { 
    return OBBVec3(x+in.x, y+in.y, z+in.z ); 
  }

  OBBVec3 operator-(OBBVec3 in) { 
    return OBBVec3(x-in.x, y-in.y, z-in.z ); 
  }
  
  OBBVec3 operator*(double in) { 
    return OBBVec3(x*in, y*in, z*in ); 
  }
  
  OBBVec3 operator/(double in) { 
    return OBBVec3(x/in, y/in, z/in ); 
  }
  
  OBBVec3 operator+=(OBBVec3 in) { 
    x+=in.x; 
    y+=in.y; 
    z+=in.z; 
    return *this; 
  }
  
  OBBVec3 operator-=(OBBVec3 in) { 
    x-=in.x; 
    y-=in.y; 
    z-=in.z; 
    return *this; 
  }
  
  OBBVec3 operator*=(double in) { 
    x*=in; 
    y*=in; 
    z*=in; 
    return *this; 
  }
  
  OBBVec3 operator/=(double in) { 
    x/=in; 
    y/=in; 
    z/=in; 
    return *this; 
  }
  
  double Dot(OBBVec3 in) { 
    return x*in.x+y*in.y+z*in.z; 
  }
  
  OBBVec3 Cross(OBBVec3 in) { 
    return OBBVec3(y*in.z-z*in.y, z*in.x-x*in.z, x*in.y-y*in.x); 
  }
  
  double Length() {
    return sqrt(x*x + y*y + z*z); 
  }

  double LengthSquared() { 
    return x*x + y*y + z*z; 
  }
  
  double Normalize() { 
    double len = Length(); 
    x/=len; 
    y/=len; 
    z/=len; 
    return len; 
  }
  
  OBBVec3 Min(OBBVec3 in) { 
    return OBBVec3(std::min(x,in.x), std::min(y,in.y), std::min(z,in.z)); 
  }
  
  OBBVec3 Max(OBBVec3 in) { 
    return OBBVec3(std::max(x,in.x), std::max(y,in.y), std::max(z,in.z)); 
  }
};

#endif
