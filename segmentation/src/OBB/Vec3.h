#ifndef DGP_VEC3_H
#define DGP_VEC3_H

#include<cmath>
#include<algorithm>

class Vec3 {
public:
	double x;
	double y;
	double z;

	Vec3( double X=0.0f, double Y=0.0f, double Z=0.0f ){
		Set(X,Y,Z);
	}

	void Set( double X, double Y, double Z ){
		x=X; y=Y; z=Z;
	}

	double &operator[]( int id ){ return *(&x+id); }

	Vec3 operator-(){ return Vec3( -x, -y, -z ); }

	Vec3 operator+( Vec3 in ){ return Vec3(x+in.x, y+in.y, z+in.z ); }
	Vec3 operator-( Vec3 in ){ return Vec3(x-in.x, y-in.y, z-in.z ); }
	Vec3 operator*( double in ){ return Vec3(x*in, y*in, z*in ); }
	Vec3 operator/( double in ){ return Vec3(x/in, y/in, z/in ); }

	Vec3 operator+=( Vec3 in ){ x+=in.x; y+=in.y; z+=in.z; return *this; }
	Vec3 operator-=( Vec3 in ){ x-=in.x; y-=in.y; z-=in.z; return *this; }
	Vec3 operator*=( double in ){ x*=in; y*=in; z*=in; return *this; }
	Vec3 operator/=( double in ){ x/=in; y/=in; z/=in; return *this; }

	double Dot( Vec3 in ){ return x*in.x+y*in.y+z*in.z; }
	Vec3  Cross( Vec3 in ){ return Vec3( y*in.z-z*in.y, z*in.x-x*in.z, x*in.y-y*in.x ); }

	double Length(){ return sqrt( x*x + y*y + z*z ); }
	double LengthSquared(){ return x*x + y*y + z*z; }
	double Normalize(){ double len = Length(); x/=len; y/=len; z/=len; return len; }

	Vec3 Min( Vec3 in ){ return Vec3( std::min(x,in.x), std::min(y,in.y), std::min(z,in.z) ); }
	Vec3 Max( Vec3 in ){ return Vec3( std::max(x,in.x), std::max(y,in.y), std::max(z,in.z) ); }
};

#endif
