#ifndef OBB_H
#define OBB_H

#include <vector>
#include "OBB_Vec3.h"
#include "eig3.h"


class OBB {
private:
  // rotation matrix for the transformation, stored as rows
  OBBVec3 m_rot[3];
  // translation component of the transformation
  OBBVec3 m_pos;
  // bounding box extents
  OBBVec3 m_ext;


  // method to set the OBB parameters which produce a box oriented according to
  // the covariance matrix C, which just containts the points pnts
  void build_from_covariance_matrix(double C[3][3], std::vector<OBBVec3> &pnts)
  {
    // extract the eigenvalues and eigenvectors from C
    double eigvec[3][3];
    double eigval[3];
    eigen_decomposition(C, eigvec, eigval);

    // find the right, up and forward vectors from the eigenvectors
    OBBVec3 r(eigvec[0][0], eigvec[1][0], eigvec[2][0]);
    OBBVec3 u(eigvec[0][1], eigvec[1][1], eigvec[2][1]);
    OBBVec3 f(eigvec[0][2], eigvec[1][2], eigvec[2][2]);
    r.Normalize(); 
    u.Normalize(); 
    f.Normalize();

    // set the rotation matrix using the eigvenvectors
    m_rot[0][0]=r.x; m_rot[0][1]=u.x; m_rot[0][2]=f.x;
    m_rot[1][0]=r.y; m_rot[1][1]=u.y; m_rot[1][2]=f.y;
    m_rot[2][0]=r.z; m_rot[2][1]=u.z; m_rot[2][2]=f.z;

    // now build the bounding box extents in the rotated frame
    OBBVec3 minim(1e10, 1e10, 1e10), maxim(-1e10, -1e10, -1e10);
    for(int i=0; i<(int)pnts.size(); i++){
      OBBVec3 p_prime(r.Dot(pnts[i]), u.Dot(pnts[i]), f.Dot(pnts[i]));
      minim = minim.Min(p_prime);
      maxim = maxim.Max(p_prime);
    }

    // set the center of the OBB to be the average of the 
    // minimum and maximum, and the extents be half of the
    // difference between the minimum and maximum
    OBBVec3 center = (maxim+minim)*0.5;
    m_pos.Set(m_rot[0].Dot(center), m_rot[1].Dot(center), m_rot[2].Dot(center));
    m_ext = (maxim-minim)*0.5;
  }

public:
  OBB(){ }

  // returns the volume of the OBB, which is a measure of
  // how tight the fit is.  Better OBBs will have smaller 
  // volumes
  double volume(){
    return 8*m_ext[0]*m_ext[1]*m_ext[2];
  }

  // constructs the corner of the aligned bounding box
  // in world space
  void get_bounding_box(OBBVec3 *p){
    OBBVec3 r(m_rot[0][0], m_rot[1][0], m_rot[2][0]);
    OBBVec3 u(m_rot[0][1], m_rot[1][1], m_rot[2][1]);
    OBBVec3 f(m_rot[0][2], m_rot[1][2], m_rot[2][2]);
    p[0] = m_pos - r*m_ext[0] - u*m_ext[1] - f*m_ext[2];
    p[1] = m_pos + r*m_ext[0] - u*m_ext[1] - f*m_ext[2];
    p[2] = m_pos + r*m_ext[0] - u*m_ext[1] + f*m_ext[2];
    p[3] = m_pos - r*m_ext[0] - u*m_ext[1] + f*m_ext[2];
    p[4] = m_pos - r*m_ext[0] + u*m_ext[1] - f*m_ext[2];
    p[5] = m_pos + r*m_ext[0] + u*m_ext[1] - f*m_ext[2];
    p[6] = m_pos + r*m_ext[0] + u*m_ext[1] + f*m_ext[2];
    p[7] = m_pos - r*m_ext[0] + u*m_ext[1] + f*m_ext[2];
  }

  // build an OBB from a vector of input points.  This
  // method just forms the covariance matrix and hands
  // it to the build_from_covariance_matrix() method
  // which handles fitting the box to the points
  void build_from_points(std::vector<OBBVec3> &pnts){
    OBBVec3 mu(0.0, 0.0, 0.0);
    double C[3][3];

    // loop over the points to find the mean point
    // location
    for(int i=0; i<(int)pnts.size(); i++){
      mu += pnts[i]/double(pnts.size());
    }

    // loop over the points again to build the 
    // covariance matrix.  Note that we only have
    // to build terms for the upper trianglular 
    // portion since the matrix is symmetric
    double cxx=0.0, cxy=0.0, cxz=0.0, cyy=0.0, cyz=0.0, czz=0.0;
    for(int i=0; i<(int)pnts.size(); i++ ){
      OBBVec3 &p = pnts[i];
      cxx += p.x*p.x - mu.x*mu.x; 
      cxy += p.x*p.y - mu.x*mu.y; 
      cxz += p.x*p.z - mu.x*mu.z;
      cyy += p.y*p.y - mu.y*mu.y;
      cyz += p.y*p.z - mu.y*mu.z;
      czz += p.z*p.z - mu.z*mu.z;
    }

    // now build the covariance matrix
    C[0][0] = cxx; C[0][1] = cxy; C[0][2] = cxz;
    C[1][0] = cxy; C[1][1] = cyy; C[1][2] = cyz;
    C[2][0] = cxz; C[2][1] = cyz; C[2][2] = czz;

    // set the OBB parameters from the covariance matrix
    build_from_covariance_matrix(C, pnts);
  }

  // builds an OBB from triangles specified as an array of
  // points with integer indices into the point array. Forms
  // the covariance matrix for the triangles, then uses the
  // method build_from_covariance_matrix() method to fit 
  // the box.  ALL points will be fit in the box, regardless
  // of whether they are indexed by a triangle or not.
  void build_from_triangles(std::vector<OBBVec3> &pnts, std::vector<int> &tris){
    double Ai, Am=0.0;
    OBBVec3 mu(0.0f, 0.0f, 0.0f), mui;
    double C[3][3];
    double cxx=0.0, cxy=0.0, cxz=0.0, cyy=0.0, cyz=0.0, czz=0.0;

    // loop over the triangles this time to find the
    // mean location
    for(int i=0; i<(int)tris.size(); i+=3){
      OBBVec3 &p = pnts[tris[i+0]];
      OBBVec3 &q = pnts[tris[i+1]];
      OBBVec3 &r = pnts[tris[i+2]];
      mui = (p+q+r)/3.0;
      Ai = (q-p).Cross(r-p).Normalize()/2.0;
      mu += mui*Ai;
      Am += Ai;

      // these bits set the c terms to Am*E[xx], Am*E[xy], Am*E[xz]....
      cxx += ( 9.0*mui.x*mui.x + p.x*p.x + q.x*q.x + r.x*r.x )*(Ai/12.0);
      cxy += ( 9.0*mui.x*mui.y + p.x*p.y + q.x*q.y + r.x*r.y )*(Ai/12.0);
      cxz += ( 9.0*mui.x*mui.z + p.x*p.z + q.x*q.z + r.x*r.z )*(Ai/12.0);
      cyy += ( 9.0*mui.y*mui.y + p.y*p.y + q.y*q.y + r.y*r.y )*(Ai/12.0);
      cyz += ( 9.0*mui.y*mui.z + p.y*p.z + q.y*q.z + r.y*r.z )*(Ai/12.0);
    }
    // divide out the Am fraction from the average position and 
    // covariance terms
    mu /= Am;
    cxx /= Am; cxy /= Am; cxz /= Am; cyy /= Am; cyz /= Am; czz /= Am;

    // now subtract off the E[x]*E[x], E[x]*E[y], ... terms
    cxx -= mu.x*mu.x; cxy -= mu.x*mu.y; cxz -= mu.x*mu.z;
    cyy -= mu.y*mu.y; cyz -= mu.y*mu.z; czz -= mu.z*mu.z;

    // now build the covariance matrix
    C[0][0]=cxx; C[0][1]=cxy; C[0][2]=cxz;
    C[1][0]=cxy; C[1][1]=cyy; C[1][2]=cyz;
    C[2][0]=cxz; C[1][2]=cyz; C[2][2]=czz;

    // set the obb parameters from the covariance matrix
    build_from_covariance_matrix(C, pnts);
  }
};

#endif
