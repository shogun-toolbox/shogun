__stringify(

__kernel void apply(__global const double * A,int A_size1,int A_size2,
		    __global const double * x,
		    __global double * y,
		    const double beta)
{
  for(unsigned int i=get_global_id(0) ; i<A_size1; i+=get_global_size(0)){
      double sum=0;
      for(unsigned int j=0 ; j<A_size2 ; ++j){
	sum+=A[i*A_size2+j]*x[j];
      }
      y[i] = sum + beta;
  }
}

)