#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <sys/time.h>
#include <immintrin.h>
#include <omp.h>
#include <math.h>



static void minmaxOMP(float const* data, int n, float* min, float* max) {

		const int nthreads = omp_get_max_threads();
		float mins[nthreads];
		float maxs[nthreads];
		const int r = n % 8;
		const int q = (n-r)/8;
		
		#pragma omp parallel 
		{
		     float local_min, local_max;

				__m256 min_clips = _mm256_load_ps(data);
				__m256 max_clips = _mm256_load_ps(data);		
										
				#pragma omp for
				for (int i = 1; i < q; i++){  
						__m256 clips = _mm256_load_ps(data + i*8);  
						max_clips = _mm256_max_ps(max_clips, clips);
						min_clips = _mm256_min_ps(min_clips, clips);      
				}
				
				local_min = min_clips[0];
				local_max = max_clips[0];
				
				for(int i = 1; i < 8; i++){
						const float imin = min_clips[i];
						const float imax = max_clips[i];
						if ( imin < local_min )
								local_min = imin;
								
						if ( imax > local_max )
								local_max = imax;	
				}
				
				mins[omp_get_thread_num()] = local_min;
				maxs[omp_get_thread_num()] = local_max;
    }
    
		*min = mins[0];
		*max = maxs[0];    
		
   	for (int i = 1; i < nthreads; i++) {
				const float imin = mins[i];
				const float imax = maxs[i];
				if (imin < *min) 
						*min = imin;

				if (imax > *max) 
						*max = imax;
		}
		
    for (int i = n-r; i < n; i++){
				const float idata = data[i];
        if ( idata > *max )
            *max = idata;
            
        else if ( idata < *min )
            *min = idata;                
    }
}

static void minmax(float const* data, int n, float* min, float* max) {

    *min = UINT_MAX;
    *max = 0;
    
    for (int i = 0; i < n; i++){
    		const float idata = data[i];
        if ( idata > *max )
            *max = idata;     
                  
        else if ( idata < *min )
            *min = idata;                           
    }
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

		
    for (int n = begin; n <= end; n+=step, c++){
    
    		naxis[c] = log2(n);
    
				float* data = _mm_malloc (n * sizeof(*data), 32); 
				
				for(int i = 0; i < n; ++i)	
				    data[i] = (unsigned) i * INT_MAX / 3; 

				struct timeval start, stop;
				float min, max;
		
				gettimeofday(&start, NULL);
				minmax(data, n, &min, &max);
				gettimeofday(&stop, NULL);
				non_opti_times[c] = (stop.tv_sec - start.tv_sec) * 1000. + (stop.tv_usec - start.tv_usec) / 1000.;

		
				gettimeofday(&start, NULL);
				minmaxOMP(data, n, &min, &max);
				gettimeofday(&stop, NULL);
				opti_times[c] = (stop.tv_sec - start.tv_sec) * 1000. + (stop.tv_usec - start.tv_usec) / 1000.;
				
				const int r = (int) 100*c/nsteps;

				if(r >= 10*prc){
						system("clear");
						printf("%d %%\n", r);
						prc++;
				}
				 
				_mm_free(data);
				
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
