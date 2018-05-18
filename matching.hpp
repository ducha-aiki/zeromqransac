/*------------------------------------------------------*/
/* Copyright 2013, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/

#ifndef MATCHING_HPP
#define MATCHING_HPP

#include "degensac/Fcustomdef.h"
#include <string>
#include <vector>

struct AffineRegion
{
  float x,y;            // subpixel, image coordinates
  float a11, a12, a21, a22;  // affine shape matrix
  float s;                   // scale
};

//verificator types
#ifdef WITH_ORSA
enum RANSAC_mode_t {LORANSAC,GR_TRUTH,LORANSACF,ORSA,GR_PLUS_RANSAC};
#else
enum RANSAC_mode_t {LORANSAC,GR_TRUTH,LORANSACF,GR_PLUS_RANSAC};
#endif

const int MODE_RANDOM = 0;

#ifdef __cplusplus
extern "C"
#endif
void FDsSym (const double *u, const double *F, double *p, int len);

//RANSAC_errors

#define MIN_POINTS 8 //threshold for symmetrical error check
#define USE_SECOND_BAD 1//uncomment if you need to use/output 1st geom.inconsistent region


struct TentativeCorresp
{
  AffineRegion first;
  AffineRegion second;
  bool isTrue;
};


struct TentativeCorrespList
{
  std::vector<TentativeCorresp> TCList;
  double H[3*3]; // by default H[i] = -1, if no H-estimation done
  TentativeCorrespList()
  {
    for (int i=0; i<9; i++)
      H[i] = -1;
  }

};


enum RANSAC_error_t {SAMPSON,SYMM_MAX,SYMM_SUM};


struct RANSACPars
{
  std::string port;
  std::string Alg;
  double duplicateDist;
  double err_threshold;
  double confidence;
  int max_samples;
  int localOptimization;
  double LAFCoef;
  double AngErrorDaniel;
  double HLAFCoef;
  RANSAC_error_t errorType;
  int doSymmCheck;
  int justMarkOutliers;
  RANSACPars()
  {
    Alg = "H";// Can be H, F or ORSA
    duplicateDist = -1;
    err_threshold = 2.0;
    confidence = 0.99;
    max_samples = 1e5;
    localOptimization = 1;
    AngErrorDaniel = -1;
    LAFCoef = 3.0;
    HLAFCoef = 10.0;
    errorType = SYMM_SUM;
    doSymmCheck = 1;
    justMarkOutliers=0;
  }
};

void AddMatchingsToList(TentativeCorrespList &tent_list, TentativeCorrespList &new_tents);
int LORANSACFiltering(TentativeCorrespList &in_corresp,
                      TentativeCorrespList &out_corresp, double *H,
                      const RANSACPars pars);
//Functions finds the inliers using LO-RANSAC and puts them into out_corresp list. Also it stores
//homography matrix H or fundamental matrix F.

#ifdef WITH_ORSA
int ORSAFiltering(TentativeCorrespList &in_corresp,
                  TentativeCorrespList &ransac_corresp,
                  double *F, const RANSACPars pars, int w, int h);
#endif
void DuplicateFiltering(TentativeCorrespList &in_corresp, const double r = 3.0, const int mode = MODE_RANDOM);
//Function does pairwise computing of the distance between ellipse centers in 1st and 2nd images.
//If distance^2 < r_sq in both images, correspondences are considered as duplicates and
//second point is deleted.

int NaiveHCheck(TentativeCorrespList &corresp,double *H,const double error);
//Performs check if the symmetrical reprojection error > given error. Returns number of "bad" points

int F_LAF_check(std::vector<TentativeCorresp> &in_matches, double *F, std::vector<TentativeCorresp> &res,
                const double affineFerror = 12.0,FDsPtr FDS1= FDsSym);
//Performs check if the full local affine frame is consistent with F-matrix.
//Error function is given in FDsPtr by user

#endif // MATCHING_HPP
