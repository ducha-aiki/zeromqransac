/*------------------------------------------------------*/
/* Copyright 2013, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/

#undef __STRICT_ANSI__

#include "../degensac/exp_ranF.h"
#include "../degensac/exp_ranH.h"
#include "ranH.h"
#include "ranF.h"

#include "matching.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>
#include <fstream>

#define DO_TRANSFER_H_CHECK


#ifdef WITH_VLFEAT
#include <kdtree.h>
#include <host.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef WITH_ORSA
#include "../orsa.h"
#endif

#define WRITE_H 1
#define VERB 0
using namespace std;

//
#ifdef __cplusplus
extern "C"
#endif
void FDsfull (const double *u, const double *F, double *p, int len);

#ifdef __cplusplus
extern "C"
#endif
void FDs (const double *u, const double *F, double *p, int len);


#ifdef __cplusplus
extern "C"
#endif
void exFDs (const double *u, const double *F, double *p, double *w, int len);

#ifdef __cplusplus
extern "C"
#endif
void FDsSym (const double *u, const double *F, double *p, int len);


#ifdef __cplusplus
extern "C"
#endif
void exFDsSym (const double *u, const double *F, double *p, double *w, int len);

void  GetEpipoles (double *F, double *e1, double *e2)
{
  cv::Mat Fmat (3,3,CV_64F,F);
  cv::Mat U,D,V;
  cv::SVDecomp(Fmat,D,U,V,4);


  e2[0] = U.at<double>(0,2) / U.at<double>(2,2);
  e2[1] = U.at<double>(1,2) / U.at<double>(2,2);
  e2[2] = 1.0;

  e1[0] = V.at<double>(0,2) / V.at<double>(2,2);
  e1[1] = V.at<double>(1,2) / V.at<double>(2,2);
  e1[2] = 1.0;

}
void GetEpipolarLine(double *e, double *pt, double *l, double &k, double &b)
{
  l[0] = e[1]*pt[2] - e[2]*pt[1];
  l[1] = e[2]*pt[0] - e[0]*pt[2];
  l[2] = e[0]*pt[1] - e[1]*pt[0];

  double x_crossx = - l[2] / l[0];
  double x_crossy = 0;
  double y_crossx = 0;
  double y_crossy = -l[2] / l[1];
  k = (y_crossx - y_crossy)/(x_crossx - x_crossy);
  b = y_crossy;
}


void GetEpipolarLineF(double *F, double *pt, double *l, double &k, double &b)
{

  l[0] = pt[0]*F[0] + pt[1]*F[3] + pt[2]*F[6];
  l[1] = pt[0]*F[1] + pt[1]*F[4] + pt[2]*F[7];
  l[2] = pt[0]*F[2] + pt[1]*F[5] + pt[2]*F[8];

  double x_crossx = - l[2] / l[0];
  double x_crossy = 0;
  double y_crossx = 0;
  double y_crossy = -l[2] / l[1];
  k = (y_crossx - y_crossy)/(x_crossx - x_crossy);
  b = y_crossy;
}
//
const double k_sigma = 3.0;

inline double distanceSq (const AffineRegion &kp1,const AffineRegion &kp2)
{
  double dx = kp1.x - kp2.x;
  double dy = kp1.y - kp2.y;
  return dx*dx + dy*dy;
}
inline void oppositeDirection (AffineRegion &kp1)
{
  kp1.a11 = - kp1.a11;
  kp1.a12 = - kp1.a12;
  kp1.a21 = - kp1.a21;
  kp1.a22 = - kp1.a22;

}

double GetAngularError(cv::Mat F, cv::Mat A, cv::Mat pt1, cv::Mat pt2)
{
  // Get the epipolar lines
  const cv::Mat l1 = F.t() * pt2;
  const cv::Mat l2 = F * pt1;

  // Their normals
  const cv::Mat n1 = (cv::Mat_<double>(2, 1) << l1.at<double>(0), l1.at<double>(1));
  const cv::Mat n2 = (cv::Mat_<double>(2, 1) << l2.at<double>(0), l2.at<double>(1));

  // Transform by A
  const cv::Mat n1_normed = n1 / norm(n1);
  const cv::Mat n2_est = -A.t() * n2;
  const cv::Mat n2_est_normed = n2_est / norm(n2_est);

  // Angular error
  return acos(n1_normed.dot(n2_est_normed));
}


int AngErrF_LAF_check(std::vector<TentativeCorresp> &in_matches, double *F,
                      std::vector<TentativeCorresp> &res,
                      const double ang_err_th, FDsPtr FDS1) {

  int n_tents = (int)in_matches.size();
  int bad_pts=0;
  std::vector<TentativeCorresp> good_matches;
  std::vector<int> good_pts(n_tents);
  for (int a=0; a<n_tents; a++)
    good_pts[a]=1; //initialization
  cv::Mat Fcv = cv::Mat_<double>(3, 3);
  int ii = 0;
  for (int i = 0; i<3 ; i++) {
      for (int j = 0; j<3 ; j++) {
          Fcv.at<double>(i,j)= F[ii];
          ii++;
        }
    }

  if (ang_err_th > 0)
    {

      std::vector<TentativeCorresp>::iterator ptr =  in_matches.begin();
      for (int l=0; l<n_tents; l++,ptr++)
        {
          cv::Mat A1(2,2,CV_64F), A2(2,2,CV_64F), A(2,2,CV_64F);
          cv::Mat pt1(3,1,CV_64F);
          cv::Mat pt2(3,1,CV_64F);
          pt1.at<double>(0,0)=  ptr->first.x;
          pt1.at<double>(1,0) =  ptr->first.y;
          pt1.at<double>(2,0) =  1.0;


          pt2.at<double>(0,0) =   ptr->second.x;
          pt2.at<double>(1,0) =  ptr->second.y;
          pt2.at<double>(2,0) =  1.0;

          A1.at<double>(0,0) = ptr->first.a11*ptr->first.s;
          A1.at<double>(0,1) = ptr->first.a12*ptr->first.s;
          A1.at<double>(1,0) = ptr->first.a21*ptr->first.s;
          A1.at<double>(1,1) = ptr->first.a22*ptr->first.s;

          A2.at<double>(0,0) = ptr->second.a11*ptr->second.s;
          A2.at<double>(0,1) = ptr->second.a12*ptr->second.s;
          A2.at<double>(1,0) = ptr->second.a21*ptr->second.s;
          A2.at<double>(1,1) = ptr->second.a22*ptr->second.s;

          A =  A2 * A1.inv();


          double curr_err = GetAngularError(Fcv, A, pt1, pt2);
          //std::cout << A << " err[rad]= " << curr_err <<  std::endl;
          if (curr_err > ang_err_th)   {
              good_pts[l]=0;
              bad_pts++;
            }
        }
      good_matches.reserve(n_tents - bad_pts);
      for (int l=0; l<n_tents; l++)
        if (good_pts[l]) good_matches.push_back(in_matches[l]);
      res = good_matches;
    }
  else res = in_matches;
  return res.size();

}

int F_LAF_check(std::vector<TentativeCorresp> &in_matches, double *F,
                std::vector<TentativeCorresp> &res,
                const double affineFerror, FDsPtr FDS1)
{
  int n_tents = (int)in_matches.size();
  int bad_pts=0;
  std::vector<TentativeCorresp> good_matches;
  std::vector<int> good_pts(n_tents);
  for (int a=0; a<n_tents; a++)
    good_pts[a]=1; //initialization

  cv::Mat Fcv = cv::Mat_<double>(3, 3);
  int ii = 0;
  for (int i = 0; i<3 ; i++) {
      for (int j = 0; j<3 ; j++) {
          Fcv.at<double>(i,j)= F[ii];
          ii++;
        }
    }
  if (affineFerror > 0)
    {

      std::vector<TentativeCorresp>::iterator ptr =  in_matches.begin();
      for (int l=0; l<n_tents; l++,ptr++)
        {
          double u[18],err[3];
          u[0] = ptr->first.x;
          u[1] = ptr->first.y;
          u[2] = 1.0;

          u[3] = ptr->second.x;
          u[4] = ptr->second.y;
          u[5] = 1.0;

          u[6] = u[0]+k_sigma*ptr->first.a12*ptr->first.s;
          u[7] = u[1]+k_sigma*ptr->first.a22*ptr->first.s;
          u[8] = 1.0;

          u[9]  = u[3]+k_sigma*ptr->second.a12*ptr->second.s;
          u[10] = u[4]+k_sigma*ptr->second.a22*ptr->second.s;
          u[11] = 1.0;

          u[12] = u[0]+k_sigma*ptr->first.a11*ptr->first.s;
          u[13] = u[1]+k_sigma*ptr->first.a21*ptr->first.s;
          u[14] = 1.0;

          u[15] = u[3]+k_sigma*ptr->second.a11*ptr->second.s;
          u[16] = u[4]+k_sigma*ptr->second.a21*ptr->second.s;
          u[17] = 1.0;

          FDS1(u,F,err,3);
          double sumErr=sqrt(err[0])+sqrt(err[1])+sqrt(err[2]);

          const bool is_good = (sumErr <= affineFerror);

            if (!is_good)
            {
              good_pts[l]=0;
              bad_pts++;
            }
        }
      good_matches.reserve(n_tents - bad_pts);
      for (int l=0; l<n_tents; l++)
        if (good_pts[l]) good_matches.push_back(in_matches[l]);
      res = good_matches;
    }
  else res = in_matches;
  return res.size();
}
int H_LAF_check(std::vector<TentativeCorresp> &in_matches,
                double *H, std::vector<TentativeCorresp> &res,
                const double affineFerror, HDsPtr HDS1)
{
  int n_tents = (int)in_matches.size();
  int bad_pts=0;
  std::vector<TentativeCorresp> good_matches;
  std::vector<int> good_pts(n_tents);
  for (int a=0; a<n_tents; a++)
    good_pts[a]=1; //initialization

  double *lin2Ptr = new double[n_tents*6], *lin;
  lin=lin2Ptr;

  if (affineFerror > 0)
    {
      std::vector<TentativeCorresp>::iterator ptr =  in_matches.begin();
      for (int l=0; l<n_tents; l++,ptr++)
        {
          double u[18],err[3];
          u[0] = ptr->first.x;
          u[1] = ptr->first.y;
          u[2] = 1.0;

          u[3] = ptr->second.x;
          u[4] = ptr->second.y;
          u[5] = 1.0;

          u[6] = u[0]+k_sigma*ptr->first.a12*ptr->first.s;
          u[7] = u[1]+k_sigma*ptr->first.a22*ptr->first.s;
          u[8] = 1.0;

          u[9]  = u[3]+k_sigma*ptr->second.a12*ptr->second.s;
          u[10] = u[4]+k_sigma*ptr->second.a22*ptr->second.s;
          u[11] = 1.0;

          u[12] = u[0]+k_sigma*ptr->first.a11*ptr->first.s;
          u[13] = u[1]+k_sigma*ptr->first.a21*ptr->first.s;
          u[14] = 1.0;

          u[15] = u[3]+k_sigma*ptr->second.a11*ptr->second.s;
          u[16] = u[4]+k_sigma*ptr->second.a21*ptr->second.s;
          u[17] = 1.0;
          HDS1(lin,u,H,err,3);

          double sumErr=sqrt(err[0] + err[1] + err[2]);
          if (sumErr > affineFerror)
            {
              good_pts[l]=0;
              bad_pts++;
            }
        }
      good_matches.reserve(n_tents - bad_pts);
      for (int l=0; l<n_tents; l++)
        if (good_pts[l]) good_matches.push_back(in_matches[l]);
      res = good_matches;
    }
  else res = in_matches;
  delete [] lin;
  return res.size();
}
void AddMatchingsToList(TentativeCorrespList &tent_list, TentativeCorrespList &new_tents)
{
  int size = (int)tent_list.TCList.size();
  unsigned int new_size = size + (int)new_tents.TCList.size();
  std::vector<TentativeCorresp>::iterator ptr =new_tents.TCList.begin();
  for (unsigned int i=size; i< new_size; i++, ptr++)
    tent_list.TCList.push_back(*ptr);
}



int LORANSACFiltering(TentativeCorrespList &in_corresp, TentativeCorrespList &ransac_corresp,double *H, const RANSACPars pars)
{
  int do_lo = pars.localOptimization;
  unsigned int i;
  unsigned int tent_size = in_corresp.TCList.size();
  int  true_size = 0;
  ransac_corresp.TCList.clear();
  int max_samples = pars.max_samples;
  if (tent_size <=20) max_samples = 1000;
  int oriented_constr = 1;
  HDsPtr HDS1;
  HDsiPtr HDSi1;
  HDsidxPtr HDSidx1;
  FDsPtr FDS1;
  exFDsPtr EXFDS1;
  switch (pars.errorType)
    {
    case SAMPSON:
      {
        HDS1 = &HDs;
        HDSi1 = &HDsi;
        HDSidx1 = &HDsidx;
        FDS1 = &FDs;
        EXFDS1 = &exFDs;
        break;
      }
    case SYMM_MAX:
      {
        HDS1 = &HDsSymMax;
        HDSi1 = &HDsiSymMax;
        HDSidx1 = &HDsSymidxMax;
        FDS1 = &FDsSym;
        EXFDS1 = &exFDsSym;
        break;
      }
    default: //case SYMM_SUM:
      {
        HDS1 = &HDsSym;
        HDSi1 = &HDsiSym;
        HDSidx1 = &HDsSymidx;
        FDS1 = &FDsSym;
        EXFDS1 = &exFDsSym;
        break;
      }
    }
  if (tent_size < MIN_POINTS)
    {
      if (VERB)  cout << tent_size << " points is not enought points to do RANSAC" << endl;
      ransac_corresp.TCList.clear();
      true_size = 0;
      return 0;
    }

  double Hloran[3*3];
  double *u2Ptr = new double[tent_size*6], *u2;
  u2=u2Ptr;
  typedef unsigned char uchar;
  unsigned char *inl2 = new uchar[tent_size];
  std::vector<TentativeCorresp>::iterator ptr1 = in_corresp.TCList.begin();
  for(i=0; i < tent_size; i++, ptr1++)
    {
      *u2Ptr =  ptr1->first.x;
      u2Ptr++;

      *u2Ptr =  ptr1->first.y;
      u2Ptr++;
      *u2Ptr =  1.;
      u2Ptr++;

      *u2Ptr =  ptr1->second.x;
      u2Ptr++;

      *u2Ptr =  ptr1->second.y;
      u2Ptr++;
      *u2Ptr =  1.;
      u2Ptr++;
    };
  if (pars.Alg == "F")
    {
      int* data_out = (int *) malloc(tent_size * 18 * sizeof(int));
      double *resids;
      int I_H = 0;
      int *Ihptr = &I_H;
      double HinF [3*3];
      exp_ransacFcustom(u2,tent_size, pars.err_threshold*pars.err_threshold,pars.confidence,
                        pars.max_samples,Hloran,inl2,data_out,do_lo,0,&resids, HinF,Ihptr,EXFDS1,FDS1, pars.doSymmCheck);
      free(resids);
      free(data_out);
      // if (VERB) std::cout << "Inliers in homography inside = " << I_H << std::endl;
    }
  else {
      int* data_out = (int *) malloc(tent_size * 18 * sizeof(int));
      double *resids;
      exp_ransacHcustom(u2, tent_size, pars.err_threshold*pars.err_threshold, pars.confidence,
                        max_samples, Hloran, inl2,4, data_out,oriented_constr ,0,&resids,HDS1,HDSi1,HDSidx1,pars.doSymmCheck);
      free(resids);
      free(data_out);
    }
  // writing ransac matchings list
  std::vector<TentativeCorresp>::iterator ptr2 = in_corresp.TCList.begin();
  if (!pars.justMarkOutliers)
    {
      for(i=0; i < tent_size; i++, ptr2++)
        {
          ptr2->isTrue=inl2[i];
          if (inl2[i])  {
              true_size++;
              ransac_corresp.TCList.push_back(*ptr2);
            }
        };
    }
  else
    {
      for(i=0; i < tent_size; i++, ptr2++)
        {
          ptr2->isTrue=inl2[i];
          if (inl2[i])  {
              true_size++;
            }
          ransac_corresp.TCList.push_back(*ptr2);
        };

    }

  delete [] u2;
  delete [] inl2;

  //Empirical checks
  if (!(pars.Alg == "F")) //H
    {
      cv::Mat Hlor(3,3,CV_64F, Hloran);
      cv::Mat Hinv(3,3,CV_64F);
      cv::invert(Hlor.t(),Hinv, cv::DECOMP_LU);
      double* HinvPtr = (double*)Hinv.data;
      int HIsNotZeros = 0;
      for (i=0; i<9; i++)
        HIsNotZeros = (HIsNotZeros || (HinvPtr[i] != 0.0));
      if (!HIsNotZeros)
        {
          ransac_corresp.TCList.clear();
          true_size = 0;
          return 0;
        }
      for (i=0; i<9; i++)
        {
          ransac_corresp.H[i]=HinvPtr[i];
          H[i] = HinvPtr[i];
        }
      ///
      TentativeCorrespList checked_corresp;

#ifdef DO_TRANSFER_H_CHECK
      int checked_numb=0;
      checked_numb = NaiveHCheck(ransac_corresp,ransac_corresp.H, 10.0); //if distance between point and reprojected point in both images <=10 px - additional check for degeneracy
      if (checked_numb < MIN_POINTS) {
          //     cerr << "Can`t get enough good points after naive check" << std::endl
          //                   <<  checked_numb << " good points out of " << ransac_corresp.TCList.size() <<std::endl;
          true_size = 0;
          ransac_corresp.TCList.clear();
        }
#endif
      H_LAF_check(ransac_corresp.TCList,Hloran,checked_corresp.TCList,3.0*pars.HLAFCoef*pars.err_threshold,&HDsSymMax);
      if (checked_corresp.TCList.size() < MIN_POINTS) {
          checked_corresp.TCList.clear();
          true_size = 0;
        }
      // std::cerr << checked_corresp.TCList.size() << " out of " << ransac_corresp.TCList.size() << " left after H-LAF-check" << std::endl;
      ransac_corresp.TCList = checked_corresp.TCList;
      true_size = checked_corresp.TCList.size();
    }
  else   //F
    {
      TentativeCorrespList checked_corresp;
      if (pars.AngErrorDaniel > 0 ) {
          AngErrF_LAF_check(ransac_corresp.TCList,Hloran,checked_corresp.TCList,pars.AngErrorDaniel,FDS1);
        } else {
          F_LAF_check(ransac_corresp.TCList,Hloran,checked_corresp.TCList,pars.LAFCoef*pars.err_threshold,FDS1);
        }
      if (checked_corresp.TCList.size() < MIN_POINTS) {
          checked_corresp.TCList.clear();
          true_size = 0;
        }
      std::cerr << checked_corresp.TCList.size() << " out of " << ransac_corresp.TCList.size() << " left after LAF-check" << std::endl;
      ransac_corresp.TCList = checked_corresp.TCList;
      true_size = checked_corresp.TCList.size();
      for (i=0; i<9; i++)
        ransac_corresp.H[i]=Hloran[i];
    }

  return true_size;
}
#ifdef WITH_ORSA
int ORSAFiltering(TentativeCorrespList &in_corresp, TentativeCorrespList &ransac_corresp,double *F, const RANSACPars pars, int w, int h)
{
  /// For LAF-check
  FDsPtr FDS1;
  switch (pars.errorType)
    {
    case SAMPSON:
      {
        FDS1 = &FDs;
        break;
      }
    case SYMM_MAX:
      {
        FDS1 = &FDsSym;
        break;
      }
    default: //case SYMM_SUM:
      {
        FDS1 = &FDsSym;
        break;
      }
    }


  ///
  unsigned int tent_size = in_corresp.TCList.size();
  ransac_corresp.TCList.clear();

  double F_tmp[9];
  if (tent_size >= MIN_POINTS)
    {
      //////// Use ORSA to filter out the incorrect matches.
      // store the coordinates of the matching points
      vector<Match> match_coor;
      match_coor.reserve(in_corresp.TCList.size());
      std::vector<TentativeCorresp>::iterator ptr1 = in_corresp.TCList.begin();
      for(int i=0; i < (int) tent_size; i++, ptr1++)
        {
          Match match1_coor;
          match1_coor.x1 = ptr1->second.x;
          match1_coor.y1 = ptr1->second.y;
          match1_coor.x2 = ptr1->first.x;
          match1_coor.y2 = ptr1->first.y;
          match_coor.push_back(match1_coor);
        }

      std::vector<float> index;

      int t_value=10000;
      int verb_value=0;
      int n_flag_value=0;
      int mode_value=2;
      int stop_value=0;
      float nfa_max = -2;
      float nfa = orsa(w, h, match_coor,index,t_value,verb_value,n_flag_value,mode_value,stop_value, F_tmp);


      // if the matching is significant, register the good matches
      if ( nfa < nfa_max )
        {
          cout << "The two images match! " << ransac_corresp.TCList.size() << " matchings are identified. log(nfa)=" << nfa << "." << endl;

          F[0] = F_tmp[0];    F[1] = F_tmp[3];    F[2] = F_tmp[6];
          F[3] = F_tmp[1];    F[4] = F_tmp[4];    F[5] = F_tmp[7];
          F[6] = F_tmp[2];    F[7] = F_tmp[5];    F[8] = F_tmp[8];
          for (int cc = 0; cc < (int) index.size(); cc++ )
            {
              ransac_corresp.TCList.push_back(in_corresp.TCList[cc]);
            }
          TentativeCorrespList checked_corresp;
          F_LAF_check(ransac_corresp.TCList,F,checked_corresp.TCList,pars.LAFCoef*pars.err_threshold,FDS1);
          if (checked_corresp.TCList.size() < MIN_POINTS)
            checked_corresp.TCList.clear();

          std::cerr << checked_corresp.TCList.size() << " out of " << ransac_corresp.TCList.size() << " left after LAF-check" << std::endl;
          ransac_corresp.TCList = checked_corresp.TCList;

        }
      else
        {
          cout << "The two images do not match. The matching is not significant: log(nfa)=" << nfa << "." << endl;
        }
    }
  else
    {
      if (VERB)  cout << tent_size << " points is not enought points to do ORSA" << endl;
      ransac_corresp.TCList.clear();
      return 0;
    }
  return ransac_corresp.TCList.size();
}
#endif

int NaiveHCheck(TentativeCorrespList &corresp,double *H,const double error)
{
  double err_sq = error*error;
  int corr_numb=0;
  int size = corresp.TCList.size();

  cv::Mat h1cv(3,3,CV_64F,H);
  cv::Mat h1inv(3,3,CV_64F);
  cv::invert(h1cv,h1inv,cv::DECOMP_LU);

  double *Hinv = (double*)h1inv.data;
  std::vector<TentativeCorresp>::iterator ptrOut = corresp.TCList.begin();
  for (int i=0; i<size; i++, ptrOut++)
    {
      double xa = (H[0]*ptrOut->first.x+H[1]*ptrOut->first.y+H[2])/(H[6]*ptrOut->first.x+H[7]*ptrOut->first.y+H[8]);
      double ya = (H[3]*ptrOut->first.x+H[4]*ptrOut->first.y+H[5])/(H[6]*ptrOut->first.x+H[7]*ptrOut->first.y+H[8]);
      //std::cout << "x=" << ptrOut->second.x << " y=" << ptrOut->second.y <<  "xa=" << xa << " ya=" << ya << std::endl;

      double d1=(ptrOut->second.x-xa)*(ptrOut->second.x-xa)+(ptrOut->second.y-ya)*(ptrOut->second.y-ya);

      xa = (Hinv[0]*ptrOut->second.x+Hinv[1]*ptrOut->second.y+Hinv[2])/(Hinv[6]*ptrOut->second.x+Hinv[7]*ptrOut->second.y+Hinv[8]);
      ya = (Hinv[3]*ptrOut->second.x+Hinv[4]*ptrOut->second.y+Hinv[5])/(Hinv[6]*ptrOut->second.x+Hinv[7]*ptrOut->second.y+Hinv[8]);
      double d2=(ptrOut->first.x-xa)*(ptrOut->first.x-xa)+(ptrOut->first.y-ya)*(ptrOut->first.y-ya);
      //std::cout << "x=" << ptrOut->first.x << " y=" << ptrOut->first.y <<  "xa=" << xa << " ya=" << ya << std::endl;

      //std::cout << "d1="<< sqrt(d1) << " d2=" << sqrt(d2) << std::endl;
      if ((d1 <=err_sq) && (d2<=(err_sq))) corr_numb++;
    }
  return corr_numb;
}


void DuplicateFiltering(TentativeCorrespList &in_corresp, const double r, const int mode)
{
  if (r <= 0) return; //no filtering
  unsigned int i,j;
  unsigned int tent_size = in_corresp.TCList.size();
  double r_sq = r*r;
  double d1_sq, d2_sq;
  vector <char> flag_unique;
  flag_unique = vector <char> (tent_size);
  for (i=0; i<tent_size; i++)
    flag_unique[i] = 1;

  std::vector<TentativeCorresp>::iterator ptr1 = in_corresp.TCList.begin();
  for(i=0; i < tent_size; i++, ptr1++)
    {
      if (flag_unique[i] == 0) continue;
      std::vector<TentativeCorresp>::iterator ptr2 = ptr1+1;
      for(j=i+1; j < tent_size; j++, ptr2++)
        {
          if (flag_unique[j] == 0) continue;
          double dx = (ptr1->first.x - ptr2->first.x);
          double dy = (ptr1->first.y - ptr2->first.y);
          d1_sq = dx*dx+dy*dy;
          if (d1_sq > r_sq)
            continue;
          dx = (ptr1->second.x - ptr2->second.x);
          dy = (ptr1->second.y - ptr2->second.y);
          d2_sq = dx*dx+dy*dy;
          if (d2_sq <= r_sq)
            flag_unique[j] = 0;
        }
    }
  TentativeCorrespList unique_list;
  unique_list.TCList.reserve(0.8*in_corresp.TCList.size());
  for (i=0; i<9; i++)
    unique_list.H[i] = in_corresp.H[i];

  for (i=0; i<tent_size; i++)
    if (flag_unique[i] == 1)
      unique_list.TCList.push_back(in_corresp.TCList[i]);

  in_corresp.TCList = unique_list.TCList;
}
