#include "mex.h"

#include <iostream>

#include <pthread.h>

#include <string.h>

#include <stdint.h>

#include <mutex>

int num_threads;

size_t* tft_indices;
size_t tft_indices_length;

double* output_data;
mwIndex* output_irs;
mwIndex* output_jcs;
size_t output_data_numel;

double* input0_data;
mwIndex* input0_irs;
mwIndex* input0_jcs;

double* input1_data;
mwIndex* input1_irs;
mwIndex* input1_jcs;

// TODO comment this mutex code, no need to depend on c++-11 for proper printing
std::mutex print_lock;

bool is_sparse;

std::pair <size_t,size_t> get_thr_output_data_start_end(int tid){
  size_t step_size = output_data_numel / num_threads;
  size_t thr_output_data_index_start = tid * step_size;
  size_t thr_output_data_index_end;
  if ( tid < (num_threads-1)){
    thr_output_data_index_end = (tid+1) * step_size;
  }else{
    thr_output_data_index_end = output_data_numel;
  }
  return std::make_pair(thr_output_data_index_start, thr_output_data_index_end);
}

void* compute_output_tensor_part_sparse(void *args){
  // TODO: implement
}



void* compute_output_tensor_part_dense(void *args){
  int tid = (intptr_t) args;

  std::pair <size_t,size_t> start_end = get_thr_output_data_start_end(tid);
  
  print_lock.lock();
  std::cout << tid << " start index " << start_end.first << " end index " << start_end.second << " output_data_numel " << output_data_numel << " num_threads " <<  num_threads << std::endl;
  print_lock.unlock();
  //std::cout << "." << std::endl;

  for ( size_t output_numel_ind=start_end.first; output_numel_ind<start_end.second; output_numel_ind++ ){
    std::cout << ".";
  }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  
  // plhs: not used

  // prhs
  // 0: degree of parallelism
  num_threads = (int) mxGetScalar(prhs[0]);
  // 1: output tensor
  // 2: input 1 tensor
  // 3: input 2 tensor

  mxArray* tft_indices_mx = mexGetVariable("global", "tft_indices");
  double* tft_indices_mx_data = (double*) mxGetData( tft_indices_mx );
  tft_indices_length = mxGetNumberOfElements(tft_indices_mx);
  tft_indices = (size_t*) malloc( sizeof(size_t) * tft_indices_length );
  //std::cout << "tft_indices class id: " << mxGetClassID(tft_indices_mx) << std::endl; // check from from MATLAB/R2014a/extern/include/matrix.h mxClassID enum
  for (int i=0; i<tft_indices_length; i++){
    tft_indices[i] = (size_t) (((double*)mxGetData((( mxGetProperty( tft_indices_mx, i, "cardinality")))))[0]);
  }

  is_sparse = false;
  for (int prhs_ind=1; prhs_ind<3; prhs_ind++){
    is_sparse = is_sparse || mxIsSparse( mxGetProperty( prhs[ prhs_ind ], 0, "data" ) );
  }

  if ( is_sparse == true ){
    for (int prhs_ind=1; prhs_ind<4; prhs_ind++){
      double* target_data;
      mwIndex* target_irs;
      mwIndex* target_jcs;
      if ( prhs_ind == 1 ){
	target_data = output_data;
	target_irs = output_irs;
	target_jcs = output_jcs;      

      }else if ( prhs_ind == 2 ){
	target_data = input0_data;
	target_irs = input0_irs;
	target_jcs = input0_jcs;

      }else if ( prhs_ind == 3 ){
	target_data = input1_data;
	target_irs = input1_irs;
	target_jcs = input1_jcs;
      }

      // TODO implement
      //mxCreateSparse(max_numel, 1, current_output_jcs[1], mxREAL);
      
      mxArray* data_array = mxGetProperty( prhs[ prhs_ind ], 0, "data" );
      target_data = (double*) mxGetData(data_array);
      target_irs = mxGetIr( data_array );
      target_jcs = mxGetJc( data_array );
      //std::cout << mxGetNumberOfElements(data_array) << std::endl;
    }

  }else{
    // TODO implement
    //output_data = mxCreateNumericArray(ndim, dims, classIDflags[ifield], mxREAL);    
  }

  pthread_t threads[num_threads];
  int rc;
  for( intptr_t i=0; i < num_threads; i++ ){
    if ( is_sparse == true ){
      rc = pthread_create(&threads[i], NULL, compute_output_tensor_part_sparse, (void *)i);
    }else{
      rc = pthread_create(&threads[i], NULL, compute_output_tensor_part_dense, (void *)i);
    }

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
