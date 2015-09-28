#include "mex.h"

#include <iostream>

#include <pthread.h>

#include <string.h>

#include <stdint.h>

void* compute_output_tensor_part(void *args){
  int tid = (intptr_t) args;
  //std::cout << tid << "." << std::endl;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  
  // plhs: not used

  // prhs
  // 0: degree of parallelism
  int num_threads = (int) mxGetScalar(prhs[0]);

  pthread_t threads[num_threads];
  int rc;
  for( intptr_t i=0; i < num_threads; i++ ){
    rc = pthread_create(&threads[i], NULL, compute_output_tensor_part, (void *)i);
    if (rc){
      std::cout << "gtp_mex: error unable to create thread " << rc << " " << strerror(rc) << std::endl;
      exit(-1);
    }
  }

  for( int i=0; i < num_threads; i++ ){
    rc = pthread_join(threads[i], NULL);
    if (rc) {
      std::cout << "gtp_mex: failed to join thread" << (long)i << " " << strerror(rc) << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}
