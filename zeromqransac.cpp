/*------------------------------------------------------*/
/* Copyright 2013, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/

#undef __STRICT_ANSI__
#include <fstream>
#include <string>
#include <iomanip>
#include <sys/time.h>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "matching.hpp"
#include "io.h"
#include <iostream>

#ifdef WITH_ORSA
#include "orsa.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include <zmq.hpp>

using namespace std;

int main(int argc, char **argv)
{
  if ((argc < 2))
    {
      std::cerr << " ************************************************************************** " << std::endl;
      std::cerr << "Usage: " << argv[0] << " config.ini port " << std::endl;
      return 1;
    }
  configs Config1;
  if (getCLIparam(Config1,argc,argv)) return 1;
  zmq::context_t context (1);
  int socket_mode = ZMQ_REP;
  zmq::socket_t   socket(context, socket_mode);
  std::string port(argv[2]);
  Config1.RANSACParam.port = "tcp://*:" + port;
  socket.bind(Config1.RANSACParam.port);
   std::cout << " Ready at port " << Config1.RANSACParam.port  << std::endl;

  while(true) {
      zmq::message_t tentatives_msg;
      socket.recv(&tentatives_msg);
      std::vector<float> inMsg(tentatives_msg.size() / sizeof(float));
      std::memcpy(inMsg.data(), tentatives_msg.data(), tentatives_msg.size());

      TentativeCorrespList tents;
      int num_tents = inMsg.size() / (2*6);
      tents.TCList.reserve(num_tents);

      std::cout << num_tents << " tents get" << std::endl;
      int count = 0;
      for (int i = 0; i < num_tents; i++){
          TentativeCorresp tc;


          tc.first.a11 = inMsg[count];
          count++;

          tc.first.a12 = inMsg[count];
          count++;

          tc.first.x = inMsg[count];
          count++;

          tc.first.a21 = inMsg[count];
          count++;

          tc.first.a22 = inMsg[count];
          count++;

          tc.first.y = inMsg[count];
          count++;

          const float scale = sqrt(fabs(tc.first.a11*tc.first.a22 - tc.first.a12*tc.first.a21));
          tc.first.s = scale;
          tc.first.a11 /= scale;
          tc.first.a12 /= scale;
          tc.first.a21 /= scale;
          tc.first.a22 /= scale;


          tc.second.a11 = inMsg[count];
          count++;

          tc.second.a12 = inMsg[count];
          count++;

          tc.second.x = inMsg[count];
          count++;

          tc.second.a21 = inMsg[count];
          count++;

          tc.second.a22 = inMsg[count];
          count++;

          tc.second.y = inMsg[count];
          count++;

          const float scale2 = sqrt(fabs(tc.second.a11*tc.second.a22 - tc.second.a12*tc.second.a21));
          tc.second.s = scale2;
          tc.second.a11 /= scale2;
          tc.second.a12 /= scale2;
          tc.second.a21 /= scale2;
          tc.second.a22 /= scale2;
          tents.TCList.push_back(tc);
        }

      TentativeCorrespList inliers;

      DuplicateFiltering(tents, Config1.RANSACParam.duplicateDist);

      int num_inl = 0;
      if ((Config1.RANSACParam.Alg == "H") || (Config1.RANSACParam.Alg == "F")) {
          num_inl = LORANSACFiltering(tents,inliers, inliers.H, Config1.RANSACParam);
        }
#ifdef WITH_ORSA
      else if (Config1.RANSACParam.Alg == "ORSA") {
          num_inl =  ORSAFiltering(tents,inliers, inliers.H, Config1.RANSACParam, 1024, 768);
        }
#endif
      else {
          std::cerr << "Unknown ransac Alg" << Config1.RANSACParam.Alg  << std::endl;
          return 1;
        }

      //do stuff
      //convert input into corerspondences
      //run ransac
      //send corrs
      //
      if (num_inl > 0) {
          std::vector<float> outMsg(num_inl*2*6);
          count = 0;
          for (int i = 0; i < num_inl; i++){

              TentativeCorresp tc = inliers.TCList[i];
              outMsg[count] = tc.first.a11*tc.first.s;
              count++;
              outMsg[count] = tc.first.a12*tc.first.s;
              count++;
              outMsg[count] = tc.first.x;
              count++;

              outMsg[count] = tc.first.a21*tc.first.s;
              count++;
              outMsg[count] = tc.first.a22*tc.first.s;
              count++;
              outMsg[count] = tc.first.y;
              count++;

              outMsg[count] = tc.second.a11*tc.second.s;
              count++;
              outMsg[count] = tc.second.a12*tc.second.s;
              count++;
              outMsg[count] = tc.second.x;
              count++;

              outMsg[count] = tc.second.a21*tc.second.s;
              count++;
              outMsg[count] = tc.second.a22*tc.second.s;
              count++;
              outMsg[count] = tc.second.y;
              count++;
//              if (i == 0) {
  //                std::cout << outMsg[0] << " " << outMsg[1] << std::endl;
    //            }
            }
    //      std::cout << num_inl << " inl sent" << std::endl;

          zmq::message_t reply(outMsg.size() * sizeof(float)) ;
          std::memcpy ((void *) reply.data (), outMsg.data(), outMsg.size() * sizeof(float) );
          socket.send (reply);
        } else {
          std::vector<float> outMsg(12);
          count = 0;
          for (int i = 0; i < 12; i++){
              outMsg[count] = 1.;
              count++;
            }
//          std::cout << " ones set as no inliers found" << std::endl;

          zmq::message_t reply(outMsg.size() * sizeof(float)) ;
          std::memcpy ((void *) reply.data (), outMsg.data(), outMsg.size() * sizeof(float));
          socket.send (reply);
        }
    }
  socket.close();
  return 0;
}


