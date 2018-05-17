//
// Created by old-ufo on 3/30/15.
//

#include "io.h"

void GetRANSACPars(RANSACPars &pars, INIReader &reader,const char* section)
{
  pars.port  = reader.GetString(section, "port", pars.port);
  pars.err_threshold = reader.GetDouble(section, "err_threshold", pars.err_threshold);
  pars.confidence = reader.GetDouble(section, "confidence", pars.confidence);
  pars.max_samples = reader.GetInteger(section, "max_samples", pars.max_samples);
  pars.localOptimization = reader.GetInteger(section, "localOptimization", pars.localOptimization);
  pars.LAFCoef = reader.GetDouble(section, "LAFcoef", pars.LAFCoef);
  pars.duplicateDist = reader.GetDouble(section, "duplicateDist", pars.duplicateDist);
  pars.AngErrorDaniel = reader.GetDouble(section, "AngErrorDaniel", pars.AngErrorDaniel);
  pars.HLAFCoef = reader.GetDouble(section, "HLAFcoef", pars.HLAFCoef);
  pars.doSymmCheck = reader.GetInteger(section, "doSymmCheck", pars.doSymmCheck);
  pars.Alg = reader.GetString(section, "Alg", pars.Alg);

  std::vector< std::string> temp_str;
  reader.GetStringVector(section, "ErrorType",temp_str);
  if (temp_str[0].compare("Sampson")==0)
    pars.errorType = SAMPSON;
  else if (temp_str[0].compare("SymmMax")==0)
    pars.errorType = SYMM_MAX;
  else //if (temp_str[0].compare("SymmSum")==0)
    pars.errorType = SYMM_SUM;
}

int getCLIparam(configs &conf1,int argc, char **argv)
{
  conf1.config_fname = argv[1];
  INIReader ConfigIni(conf1.config_fname);
  GetRANSACPars(conf1.RANSACParam,ConfigIni);
  return 0;
}
