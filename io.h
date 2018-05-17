//
// Created by Dmytro Mishkin on 3/30/15.
//
#ifndef MODS_NEW_IO_MODS_H
#define MODS_NEW_IO_MODS_H


#include "matching.hpp"
#include "inih/cpp/INIReader.h"



struct configs
{
  int n_threads;
  std::string config_fname;
  RANSACPars RANSACParam;
  configs()
  {
    config_fname = "config.ini";
    n_threads = 1;

  }
};

void GetRANSACPars(RANSACPars &pars, INIReader &reader,const char* section="RANSAC");
int getCLIparam(configs &conf1,int argc, char **argv);


#endif //MODS_NEW_IO_MODS_H
