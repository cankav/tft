#include "mex.h"
#include <iostream>
#include <pthread.h>
#include <string.h>
#include <stdint.h>
#include <mutex> // for cout lock
#include <math.h>

int num_threads;

size_t* tft_indices_cardinalities;
size_t* tft_indices_ids;
size_t tft_indices_length;
mxArray* tft_indices_mx;

double* output_data;
mxArray* output_data_mx;
mwIndex* output_irs;
mwIndex* output_jcs;
size_t output_data_numel;
size_t output_data_numel_nzmax;
size_t* output_index_cardinalities;
size_t* output_indices_full_cardinality;
size_t* output_indices_full_strides;
size_t output_indices_length;
mxArray* output_indices_mx;

double* input0_data;
mwIndex* input0_irs;
mwIndex* input0_jcs;
size_t input0_data_numel;
size_t* input0_indices_full_cardinality;
size_t* input0_indices_full_strides;

double* input1_data;
mwIndex* input1_irs;
mwIndex* input1_jcs;
size_t input1_data_numel;
size_t* input1_indices_full_cardinality;
size_t* input1_indices_full_strides;

size_t* contraction_index_inds; //indexes tft_indices
size_t contraction_index_inds_length;

// TODO comment this mutex code, no need to depend on c++-11 for proper printing
std::mutex print_lock;

bool is_sparse;
bool is_sparse_input0;
bool is_sparse_input1;

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

double get_tensor_data_by_full_index_configuration_dense(double* tensor_data, size_t* index_configuration, size_t* tensor_indices_full_strides, size_t tensor_data_numel){
  size_t tensor_numel_index = 0;
  for (int tft_indices_ind=0; tft_indices_ind<tft_indices_length; tft_indices_ind++){
    tensor_numel_index += index_configuration[tft_indices_ind] * tensor_indices_full_strides[tft_indices_ind];
  }

  if ( tensor_numel_index < 0 || tensor_numel_index >= tensor_data_numel ){
    std::cout << "ERROR: get_tensor_data_by_index_configuration_dense tensor_numel_index " << tensor_numel_index << " can not be smaller than zero or greater than tensor_data_numel " << tensor_data_numel << std::endl;
    return 0;
  }else{
    return tensor_data[tensor_numel_index];
  }
}

void compute_output_tensor_part_dense_helper(size_t* output_full_index_configuration, size_t output_numel_index, size_t increment_index_ind=0){
  for ( size_t contraction_index_value=0;
	contraction_index_value<tft_indices_cardinalities[contraction_index_inds[increment_index_ind]];
	contraction_index_value++ ){

    output_full_index_configuration[ contraction_index_inds[increment_index_ind] ] = contraction_index_value;
    
    if ( increment_index_ind == (contraction_index_inds_length-1) ){
      output_data[output_numel_index] +=
	get_tensor_data_by_full_index_configuration_dense(input0_data, output_full_index_configuration, input0_indices_full_strides, input0_data_numel) *
	get_tensor_data_by_full_index_configuration_dense(input1_data, output_full_index_configuration, input1_indices_full_strides, input1_data_numel);
    }else{
      compute_output_tensor_part_dense_helper( output_full_index_configuration, output_numel_index, increment_index_ind+1 );
    }
  }
}

void* compute_output_tensor_part_dense(void *args){
  int tid = (intptr_t) args;

  std::pair <size_t,size_t> start_end = get_thr_output_data_start_end(tid);
  
  // print_lock.lock();
  // std::cout << tid << " start index " << start_end.first << " end index " << start_end.second << " output_data_numel " << output_data_numel << " num_threads " <<  num_threads << std::endl;
  // print_lock.unlock();
  // std::cout << "." << std::endl;

  size_t* output_full_index_configuration = (size_t*) calloc( tft_indices_length, sizeof(size_t) );
  //int loop_count = 0;
  for ( size_t output_numel_ind=start_end.first; output_numel_ind<start_end.second; output_numel_ind++ ){
    //print_lock.lock();
    // calculate output_full_index_configuration for output_numel_ind
    // for s_1 = 2, s_2 = 3, s_3 = 4
    // x = 0:23
    // [x ; mod(floor(x/12),2); mod(floor(x / 4), 3); mod(floor(x/1),4) ]'  <- data order incremented from rightmost tft_index
    // [x ; mod(floor(x/1),2) ; mod(floor(x / 2), 3);  mod(floor(x/6),4)]'  <- data order incremented from leftmost tft_index (MATLAB)
    //size_t right_hand_inds_step_divider = 1;
    size_t left_hand_inds_step_divider = 1;
    size_t output_numel_index = 0;
    for( size_t tft_indices_ind=0; tft_indices_ind<tft_indices_length; tft_indices_ind++ ){
      if ( output_indices_full_strides[tft_indices_ind] > 0 ){
	output_full_index_configuration[tft_indices_ind] = ((size_t)floor(output_numel_ind / left_hand_inds_step_divider)) % tft_indices_cardinalities[tft_indices_ind];
	//right_hand_inds_step_divider *= tft_indices_cardinalities[tft_indices_ind];
	left_hand_inds_step_divider *= tft_indices_cardinalities[tft_indices_ind];

	//std::cout << "SLM output_indices_full_strides[" << tft_indices_ind << "] " << std::endl;
	//std::cout << "output_full_index_configuration[" << tft_indices_ind << "]\t" << output_full_index_configuration[tft_indices_ind]  << " output_indices_full_strides[tft_indices_ind] \t" << output_indices_full_strides[tft_indices_ind] << " output_numel_index \t" << output_numel_index << std::endl;
	// //std::cout << "WTF" << std::endl;

	output_numel_index += output_full_index_configuration[tft_indices_ind] * output_indices_full_strides[tft_indices_ind];
      }
    }

    //print_lock.unlock();

    // print_lock.lock();
    // //std::cout << "output_numel_ind " << output_numel_ind;
    // for( size_t tft_indices_ind=0; tft_indices_ind<tft_indices_length; tft_indices_ind++ ){
    //   //std::cout << " " << output_full_index_configuration[tft_indices_ind];
    // }
    // //std::cout << std::endl;
    // print_lock.unlock();
    // print_lock.lock();
    // //std::cout << "output_numel_index " << output_numel_index << std::endl;
    // print_lock.unlock();

    //std::cout << "contraction_index_inds_length " << contraction_index_inds_length << std::endl;
    // //std::cout << "output_data " << output_data << std::endl;
    if ( contraction_index_inds_length == 0 ){
      // no contraction, just multiply and store result
      // TODO: test
      output_data[output_numel_index] =
	get_tensor_data_by_full_index_configuration_dense(input0_data, output_full_index_configuration, input0_indices_full_strides, input0_data_numel) *
	get_tensor_data_by_full_index_configuration_dense(input1_data, output_full_index_configuration, input1_indices_full_strides, input1_data_numel);

    }else{
      // loop for each combination of contraction indexes' values and store result
      output_data[output_numel_index] = 0;
      //std::cout << "output_data[" << output_numel_index << "] is " << output_data[output_numel_index] << " " << output_data[8] << std::endl;
      
      compute_output_tensor_part_dense_helper(output_full_index_configuration, output_numel_index);
    }
    //std::cout << "WTF2" << std::endl;
    // loop_count++;
    // if( loop_count == 2 )
    //print_lock.unlock();
  }

}

void init_tensor_meta_data( size_t** target_indices_full_cardinality, size_t* target_data_numel, size_t** target_indices_full_strides, const mxArray* input_mxArray ){
  *target_data_numel = 1;
  *target_indices_full_cardinality = (size_t*) malloc( sizeof(size_t) * tft_indices_length );
  *target_indices_full_strides = (size_t*) malloc( sizeof(size_t) * tft_indices_length );

  mxArray* target_indices_mx = mxGetProperty( input_mxArray, 0, "indices" );
  size_t target_indices_length = mxGetNumberOfElements(target_indices_mx);
  size_t current_stride = 1;
  for (int tft_indices_ind=0; tft_indices_ind<tft_indices_length; tft_indices_ind++){
    bool found = false;
    for ( size_t target_indices_ind=0; target_indices_ind<target_indices_length; target_indices_ind++ ){
      mxArray* prop_id = mxGetProperty( mxGetCell(target_indices_mx, target_indices_ind), 0, "id");
      size_t target_index_id = (size_t) ( ((double*)mxGetData(prop_id))[0] );
      if ( tft_indices_ids[tft_indices_ind] == target_index_id ){
	found = true;
	break;
      }
    }

    if ( found == true ){
      size_t current_cardinality = (size_t) (((double*)mxGetData((( mxGetProperty( tft_indices_mx, tft_indices_ind, "cardinality")))))[0]);
      (*target_indices_full_cardinality)[tft_indices_ind] = current_cardinality;
      *target_data_numel *= (*target_indices_full_cardinality)[tft_indices_ind];
      (*target_indices_full_strides)[tft_indices_ind] = current_stride; // TODO: check data access order - stride order
      current_stride *= current_cardinality;
    }else{
      (*target_indices_full_cardinality)[tft_indices_ind] = 0;
      (*target_indices_full_strides)[tft_indices_ind] = 0;
    }
    //std::cout << "target_indices_full_cardinality[tft_indices_ind] " << (*target_indices_full_cardinality)[tft_indices_ind] << std::endl;
  }  
}

void init_dense_tensor( double** target_data, size_t** target_indices_full_cardinality, size_t* target_data_numel, size_t** target_indices_full_strides, const mxArray* input_mxArray){
  mxArray* data_array = mxGetProperty( input_mxArray, 0, "data" );
  *target_data = (double*) mxGetData(data_array);
  init_tensor_meta_data( target_indices_full_cardinality, target_data_numel, target_indices_full_strides, input_mxArray );
}

void init_sparse_tensor(double** target_data, size_t** target_indices_full_cardinality, size_t* target_data_numel, size_t** target_indices_full_strides, const mxArray* input_mxArray, mwIndex** target_irs, mwIndex** target_jcs){
  mxArray* data_array_mx = mxGetProperty( input_mxArray, 0, "data" );
  *target_irs = mxGetIr( data_array_mx );
  *target_jcs = mxGetJc( data_array_mx );
  init_tensor_meta_data( target_indices_full_cardinality, target_data_numel, target_indices_full_strides, input_mxArray );
}

void init_dense_output_tensor(const mxArray* target_mxArray, size_t** target_indices_full_cardinality, size_t* target_data_numel, size_t** target_indices_full_strides, const mxArray* input_mxArray ){
  output_indices_mx = mxGetProperty( target_mxArray, 0, "indices" );
  output_indices_length = mxGetNumberOfElements( output_indices_mx );
  //std::cout << "SLM output_indices_length " << output_indices_length << std::endl;
  mwSize* output_data_array_cardinalities_size_dims = (mwSize*) malloc( sizeof(mwSize) * tft_indices_length );
  for ( int i=0; i<tft_indices_length; i++ )
    output_data_array_cardinalities_size_dims[i] = 1;
  mxArray* output_data_array_cardinalities_mx = mxCreateNumericArray(tft_indices_length, output_data_array_cardinalities_size_dims, mxDOUBLE_CLASS, mxREAL);
  mwSize* output_data_array_cardinalities = (mwSize*) mxGetData(output_data_array_cardinalities_mx);
  size_t output_data_array_cardinalities_index = 0;
  for ( size_t tft_indices_ind=0; tft_indices_ind<tft_indices_length; tft_indices_ind++ ){
    //std::cout << "\nSLM tft_indices_ind " << tft_indices_ind << std::endl;
    bool found = false;
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
	output_data_array_cardinalities[output_data_array_cardinalities_index] = tft_indices_cardinalities[tft_indices_ind];
	//std::cout << "SLM output_data_array_cardinalities_index " << output_data_array_cardinalities_index << std::endl;
	output_data_array_cardinalities_index++;
	found = true;
	break;
      }
      //std::cout << "SLM DONE" << std::endl;
    }
    if ( found == false ){
      // dummy dimension (due to Matlab indexing compatability)
      output_data_array_cardinalities[output_data_array_cardinalities_index] = 1;
    }
  }
  output_data_mx = mxCreateNumericArray(tft_indices_length, output_data_array_cardinalities, mxDOUBLE_CLASS, mxREAL);
  output_data = (double*) mxGetData(output_data_mx);
  init_tensor_meta_data(target_indices_full_cardinality, target_data_numel, target_indices_full_strides, input_mxArray);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  //std::cout << "SLM output_indices_full_cardinality BEFORE " << output_indices_full_cardinality << std::endl;

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

  tft_indices_mx = mexGetVariable("global", "tft_indices");
  tft_indices_length = mxGetNumberOfElements(tft_indices_mx);
  tft_indices_cardinalities = (size_t*) malloc( sizeof(size_t) * tft_indices_length );
  tft_indices_ids = (size_t*) malloc( sizeof(size_t) * tft_indices_length );
  //std::cout << "tft_indices class id: " << mxGetClassID(tft_indices_mx) << std::endl; // check from from MATLAB/R2014a/extern/include/matrix.h mxClassID enum

  for (int i=0; i<tft_indices_length; i++){
    tft_indices_cardinalities[i] = (size_t) (((double*)mxGetData((( mxGetProperty( tft_indices_mx, i, "cardinality")))))[0]);
    tft_indices_ids[i] = (size_t) (((double*)mxGetData((( mxGetProperty( tft_indices_mx, i, "id")))))[0]);
  }

  //std::cout << "SLM output_indices_full_cardinality AFTER " << output_indices_full_cardinality << std::endl;
  is_sparse = false;
  for (int prhs_ind=output_tensor_prhs_index; prhs_ind<=input1_tensor_prhs_index; prhs_ind++){
    bool is_tensor_sparse = mxIsSparse( mxGetProperty( prhs[ prhs_ind ], 0, "data" ) );
    is_sparse = is_sparse || is_tensor_sparse;
    if( prhs_ind == input0_tensor_prhs_index ){
      is_sparse_input0 = is_tensor_sparse;
    }else if( prhs_ind == input1_tensor_prhs_index ){
      is_sparse_input1 = is_tensor_sparse;
    }
  }

  //std::cout << "is_sparse: " << is_sparse << std::endl;
  if ( is_sparse == true ){
    // sparse init
    output_data_numel_nzmax = output_data_numel * 0.2; // TODO: how to set nzmax value?
    mxArray* output_data_mx = mxCreateSparse(output_data_numel, 1, output_data_numel_nzmax, mxREAL);
    output_data = mxGetPr(output_data_mx);

    if ( is_sparse_input0 == true ){
      init_sparse_tensor(&input0_data, &input0_indices_full_cardinality, &input0_data_numel, &input0_indices_full_strides, prhs[ input0_tensor_prhs_index ], &input0_irs, &input0_jcs);
    }else {
      init_dense_tensor(&input0_data, &input0_indices_full_cardinality, &input0_data_numel, &input0_indices_full_strides, prhs[ input0_tensor_prhs_index ]);
    }

    if ( is_sparse_input1 == true ){
      init_sparse_tensor(&input1_data, &input1_indices_full_cardinality, &input1_data_numel, &input1_indices_full_strides, prhs[ input1_tensor_prhs_index ], &input1_irs, &input1_jcs);
    }else {
      init_dense_tensor(&input1_data, &input1_indices_full_cardinality, &input1_data_numel, &input1_indices_full_strides, prhs[ input1_tensor_prhs_index ]);
    }

  }else{
    // dense init

    init_dense_output_tensor(prhs[output_tensor_prhs_index], &output_indices_full_cardinality, &output_data_numel, &output_indices_full_strides, prhs[ output_tensor_prhs_index ]);
    init_dense_tensor(&input0_data, &input0_indices_full_cardinality, &input0_data_numel, &input0_indices_full_strides, prhs[ input0_tensor_prhs_index ]);
    init_dense_tensor(&input1_data, &input1_indices_full_cardinality, &input1_data_numel, &input1_indices_full_strides, prhs[ input1_tensor_prhs_index ]);

    //print_lock.lock();
    //for (size_t i =0; i<tft_indices_length; i++)
      //std::cout << "output_data_array_cardinalities[" << i << "] \t" << output_data_array_cardinalities[i] << std::endl;
    //std::cout << "output data numel " << mxGetNumberOfElements(output_data_mx) << std::endl;
    //print_lock.unlock();
    
  }
  
  // generate contraction_index_inds
  contraction_index_inds = (size_t*) calloc( tft_indices_length, sizeof(size_t) );
  contraction_index_inds_length = 0;
  for ( size_t tft_indices_ind=0; tft_indices_ind<tft_indices_length; tft_indices_ind++ ){
    bool is_contraction_index = true;
    // check if index appears in output_index
    for ( size_t output_indices_ind=0; output_indices_ind<output_indices_length; output_indices_ind++ ){
      mxArray* prop_id = mxGetProperty( mxGetCell(output_indices_mx, output_indices_ind), 0, "id");
      size_t output_index_id = (size_t) ( ((double*)mxGetData(prop_id))[0] );
      if ( tft_indices_ids[tft_indices_ind] == output_index_id ){
	// index appears in output tensor -> not contraction index
	is_contraction_index = false;
	break;
      }
    }
    if ( is_contraction_index == true ){
      contraction_index_inds[contraction_index_inds_length] = tft_indices_ind;
      contraction_index_inds_length++;
    }
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

  mxSetProperty( prhs[ output_tensor_prhs_index ], 0, "data", output_data_mx );
}
