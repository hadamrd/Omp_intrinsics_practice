/*
En rÃ©utilisant les fonctions ``std::sort``, ``std::merge`` et Ã  l'aide de directives OpenMP, Ã©crivez
une version parallÃ¨les d'un algorithme de tri de nombres entiers.

Dans un premier temps, vous pourrez parallÃ©liser uniquement le tri, avec un
``merge`` sÃ©quentiel. Motivez l'utilisation d'un seuil Ã  partir duquel il
faudrait basculer sur l'algorithme sÃ©quentiel.  Quelle est alors la complexitÃ©
de l'algo parallÃ¨le ?  Essayez ensuite de parallÃ©liser le ``merge``.

Comparez vos rÃ©sultats en utilisant des tableaux d'entiers de taille
diffÃ©rente. Comment Ã©volue le nombre d'octets traitÃ©s par seconde ?


.. code-block:: c++
*/
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>       
#include <chrono>     

#define SWAP(a,b) {double temp = a; a = b; b = temp;}

using namespace std;

void mmerge( double a[ ], double tmp_array[ ], int size) {

    int tmp_iter = 0, l_iter = 0, r_iter = size/2;

    while( ( l_iter < size/2 ) && ( r_iter < size ) )

        if( a[l_iter] <= a[r_iter] )
            tmp_array[tmp_iter++] = a[l_iter++];
        else
            tmp_array[tmp_iter++] = a[r_iter++];
		
	copy(a + l_iter, a + size/2, tmp_array + tmp_iter);
	copy(a + r_iter, a + size, tmp_array + tmp_iter + size/2 - l_iter);
	copy(tmp_array, tmp_array+size, a);

}

void mergesort_serial(double arr[], int size, double tmp[]){
	
	if ( size < 2 ) return;
	else if ( size == 2 and arr[0] > arr[1]){
		SWAP(arr[0],arr[1]);
		return;
	}
	else{
		mergesort_serial(arr, size/2, tmp);
		mergesort_serial(arr + size/2, size - size/2, tmp + size/2);
		mmerge(arr, tmp, size);
	}
}

void mergesort_parallel_omp(double arr[], int size, double tmp[], int nthreads) 
{ 

	if ( nthreads == 1){
		mergesort_serial(arr, size, tmp);
	}
	else{		
		const int mid = size/2;
		#pragma omp parallel sections
		{
			omp_set_num_threads(2);
			
			#pragma omp section
			mergesort_parallel_omp(arr, mid, tmp, nthreads/2);
			
			#pragma omp section
			mergesort_parallel_omp(arr + mid, size - mid, tmp + mid, nthreads - nthreads/2);
		}				
		mmerge( arr, tmp, size);
	}
}

void shuffled_list(double *arr, int size){
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	for(int i = 0; i < size; ++i)
		arr[i] = numeric_limits<int>::max() - size + i + 1;
	shuffle (arr, arr + size, default_random_engine(seed));
}

int main(int argc, char**argv) {

		if ( argc < 4 ){
				printf("wrong number of arguments : 3 args expected, %d given.\n", argc-1);
		  	return -1;
		}
		  
		int begin = pow(2, atoi(argv[1])); 
		int end = pow(2, atoi(argv[2]));
		int nsteps = atoi(argv[3]); 
		int step = ceil( (end - begin + 1) / nsteps );
		 
		float non_opti_times[nsteps];
		float opti_times[nsteps];
		float naxis[nsteps];
		int c = 0;
		int prc = 1;

		double start, stop, T1, T2;
		
    for (int n = begin; n <= end; n+=step, c++){

		naxis[c] = log2(n);
		double *data1 = new double[n];
		double *data2 = new double[n];
    double *temp = new double[n];
		int num_threads;
		omp_set_nested(1);
		#pragma omp parallel
		{
			#pragma omp master
			{
				num_threads = omp_get_num_threads();
			}
		}
	
		shuffled_list(data1, n);
		start = omp_get_wtime();
		mergesort_parallel_omp(data1, n, temp, num_threads);
		stop = omp_get_wtime();
		opti_times[c] = stop - start;

		shuffled_list(data2, n);
		start = omp_get_wtime();
		mergesort_serial(data2, n, temp);
		stop = omp_get_wtime();
		volatile __attribute__((unused)) double anchor = data2[n/2]; // why that?
		non_opti_times[c] = stop - start;
	
	  }
	
		FILE *f = fopen("results.csv", "wb");		
		
		fprintf(f, "%f ", naxis[0]);
		for( int i = 1; i < c; i++)
    		fprintf(f, ",%f ", naxis[i]);
    
    fprintf(f, "\n");

		fprintf(f, "%f ", non_opti_times[0]);
		for( int i = 1; i < c; i++)
    		fprintf(f, ",%f ", non_opti_times[i]);
    		
    fprintf(f, "\n");

    fprintf(f, "%f ", opti_times[0]);
    for( int i = 1; i < c; i++)
    		fprintf(f, ",%f ", opti_times[i]);
    		
    fclose(f);
    
    system("python plotResultScript.py");
    return 0;
}
