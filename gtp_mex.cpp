#include "mex.h"

#include <iostream>

#include <pthread.h>

#include <string.h>

#include <stdint.h>

#include <mutex>

int num_threads;

size_t* tft_indices_cardinalities;
size_t* tft_indices_ids;
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
  
  //print_lock.lock();
  //std::cout << tid << " start index " << start_end.first << " end index " << start_end.second << " output_data_numel " << output_data_numel << " num_threads " <<  num_threads << std::endl;
  //print_lock.unlock();
  //std::cout << "." << std::endl;

  for ( size_t output_numel_ind=start_end.first; output_numel_ind<start_end.second; output_numel_ind++ ){
    
  }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  
  // plhs: not used

  // prhs
  // 0: degree of parallelism
  num_threads = (int) mxGetScalar(prhs[0]);
  // 1: output tensor
  const int output_tensor_prhs_index = 1;
  // 2: input 1 tensor
  const int input0_tensor_prhs_index = 2;
  // 3: input 2 tensor
  const int input1_tensor_prhs_index = 3;

  mxArray* tft_indices_mx = mexGetVariable("global", "tft_indices");
  tft_indices_length = mxGetNumberOfElements(tft_indices_mx);
  tft_indices_cardinalities = (size_t*) malloc( sizeof(size_t) * tft_indices_length );
  tft_indices_ids = (size_t*) malloc( sizeof(size_t) * tft_indices_length );
  //std::cout << "tft_indices class id: " << mxGetClassID(tft_indices_mx) << std::endl; // check from from MATLAB/R2014a/extern/include/matrix.h mxClassID enum
  for (int i=0; i<tft_indices_length; i++){
    tft_indices_cardinalities[i] = (size_t) (((double*)mxGetData((( mxGetProperty( tft_indices_mx, i, "cardinality")))))[0]);
    tft_indices_ids[i] = (size_t) (((double*)mxGetData((( mxGetProperty( tft_indices_mx, i, "id")))))[0]);
  }

  is_sparse = false;
  for (int prhs_ind=output_tensor_prhs_index; prhs_ind<=input1_tensor_prhs_index; prhs_ind++){
    is_sparse = is_sparse || mxIsSparse( mxGetProperty( prhs[ prhs_ind ], 0, "data" ) );
  }

  if ( is_sparse == true ){
    // TODO implement
    //mxCreateSparse(max_numel, 1, current_output_jcs[1], mxREAL);

    for (int prhs_ind=output_tensor_prhs_index; prhs_ind<=input1_tensor_prhs_index; prhs_ind++){
      double* target_data;
      mwIndex* target_irs;
      mwIndex* target_jcs;
      if ( prhs_ind == output_tensor_prhs_index ){
	target_data = output_data;
	target_irs = output_irs;
	target_jcs = output_jcs;

      }else if ( prhs_ind == input0_tensor_prhs_index ){
	target_data = input0_data;
	target_irs = input0_irs;
	target_jcs = input0_jcs;

      }else if ( prhs_ind == input1_tensor_prhs_index ){
	target_data = input1_data;
	target_irs = input1_irs;
	target_jcs = input1_jcs;
      }
      
      mxArray* data_array = mxGetProperty( prhs[ prhs_ind ], 0, "data" );
      target_data = (double*) mxGetData(data_array);
      target_irs = mxGetIr( data_array );
      target_jcs = mxGetJc( data_array );
      //std::cout << mxGetNumberOfElements(data_array) << std::endl;
    }

  }else{
    output_data_numel = 1;
    size_t ndim = tft_indices_length;
    mxArray* output_indices_mx = mxGetProperty( prhs[ output_tensor_prhs_index ], 0, "indices" );
    size_t output_indices_length = mxGetNumberOfElements( output_indices_mx );
    //std::cout << "SLM output_indices_length" << output_indices_length << std::endl;
    mwSize* output_data_array_cardinalities_size = (mwSize*) malloc( sizeof(mwSize) * tft_indices_length );
    for ( int i=0; i<tft_indices_length; i++ )
      output_data_array_cardinalities_size[i] = 1;
    mxArray* output_data_array_cardinalities_mx = mxCreateNumericArray(tft_indices_length, output_data_array_cardinalities_size, mxDOUBLE_CLASS, mxREAL);
    mwSize* output_data_array_cardinalities = (mwSize*) mxGetData(output_data_array_cardinalities_mx);
    for ( size_t tft_indices_ind=0; tft_indices_ind<ndim; tft_indices_ind++ ){
      //std::cout << "\nSLM tft_indices_ind " << tft_indices_ind << std::endl;
      for ( size_t output_indices_ind=0; output_indices_ind<output_indices_length; output_indices_ind++ ){
	//std::cout << "SLM output_indices_ind " << output_indices_ind << std::endl;
	mxArray* prop_id = mxGetProperty( mxGetCell(output_indices_mx, output_indices_ind), 0, "id");
	//std::cout << "SLM prop_id " << prop_id << std::endl;
	size_t output_index_id = (size_t) ( ((double*)mxGetData(prop_id))[0] );
	//std::cout << "SLM output_index_id " << output_index_id << std::endl;
	if ( tft_indices_ids[tft_indices_ind] == output_index_id ){
	  //std::cout << "SLM YES tft_indices_ind " << tft_indices_ind << std::endl;
	  //std::cout << "SLM output_data_array_cardinalities[tft_indices_ind] " <<  output_data_array_cardinalities[tft_indices_ind] << std::endl;
	  //std::cout << "SLM tft_indices_cardinalities[tft_indices_ind] " <<  tft_indices_cardinalities[tft_indices_ind] << std::endl;
	  output_data_array_cardinalities[tft_indices_ind] = tft_indices_cardinalities[tft_indices_ind];
	  output_data_numel *= tft_indices_cardinalities[tft_indices_ind];
	}else{
	  //std::cout << "SLM NO" << std::endl;
	  //std::cout << "SLM output_data_array_cardinalities[tft_indices_ind] " <<  output_data_array_cardinalities[tft_indices_ind] << std::endl;
	  //output_data_array_cardinalities[tft_indices_ind] = 1;
	}
	//std::cout << "SLM DONE" << std::endl;
      }
    }

    // for (size_t i =0; i<ndim; i++){
    //   print_lock.lock();
    //   std::cout << output_data_array_cardinalities[i] << std::endl;
    //   print_lock.unlock();
    // }
    mxSetProperty( prhs[ output_tensor_prhs_index ], 0, "data", mxCreateNumericArray(ndim, output_data_array_cardinalities, mxDOUBLE_CLASS, mxREAL) );
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
