#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <sys/time.h>
#include <immintrin.h>
#include <omp.h>
#include <math.h>

static void minmaxAVX(float const* data, int n, float* min, float* max) {
    
    int r = n % 8 ;
    
		__m256 min_clips = _mm256_load_ps(data);
		__m256 max_clips = _mm256_load_ps(data);
		
		for (int i = 8; i < n-r; i+=8){  
				__m256 clips = _mm256_load_ps(data + i);  
				max_clips = _mm256_max_ps(max_clips, clips);
				min_clips = _mm256_min_ps(min_clips, clips);      
		}
	
		for(int i = 0; i < 8; i++){
				if ( min_clips[i] < *min )
						*min = min_clips[i];
						
				if ( max_clips[i] > *max )
						*max = max_clips[i];	
		}

    for (int i = n-r; i < n; i++){
				const float idata = data[i];
				   
        if ( idata > *max )
            *max = idata;
            
        else if ( idata < *min )
            *min = idata;                
    }
}
# if 0
static void minmaxSSE(float const* data, int n, float* min, float* max) {
    
    int r = n % 4 ;
    
		__m128 min_clips = _mm_load_ps(data);
		__m128 max_clips = _mm_load_ps(data);
		
		for (int i = 4; i < n-r; i+=4){  
				__m128 clips = _mm_load_ps(data + i);  
				max_clips = _mm_max_ps(max_clips, clips);
				min_clips = _mm_min_ps(min_clips, clips);      
		}
	
		for(int i = 0; i < 4; i++){
				if ( min_clips[i] < *min )
						*min = min_clips[i];
		
				if ( max_clips[i] > *max )
						*max = max_clips[i];	
		}

    for (int i = n-r; i < n; i++){
        if ( data[i] > *max )
            *max = data[i];
            
        if ( data[i] < *min )
            *min = data[i];                
    }
}
#endif
static void minmax(float const* data, int n, float* min, float* max) {

    *min = UINT_MAX;
    *max = 0;
    
    for (int i = 0; i < n; i++){
# if 0
        if ( data[i] > *max )
            *max = data[i];           
        if ( data[i] < *min )
            *min = data[i];  
#else
        *max= fmaxf(*max, data[i]);                         
        *min= fminf(*min, data[i]);
#endif
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
				    data[i] = (unsigned) i * (INT_MAX / 3); 

				struct timeval start, stop;
				float min, max;
		
				gettimeofday(&start, NULL);
				minmax(data, n, &min, &max);
				volatile float f = min + max;
				gettimeofday(&stop, NULL);
				non_opti_times[c] = (stop.tv_sec - start.tv_sec) * 1000. + (stop.tv_usec - start.tv_usec) / 1000.;

		
				gettimeofday(&start, NULL);
				minmaxAVX(data, n, &min, &max);
				volatile float fp = min + max;
				gettimeofday(&stop, NULL);
				opti_times[c] = (stop.tv_sec - start.tv_sec) * 1000. + (stop.tv_usec - start.tv_usec) / 1000.;
				
				const int r = (int) 100*c/nsteps;

				if(r >= prc){
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
