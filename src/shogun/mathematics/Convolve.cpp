#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <time.h>

using namespace std;

int MATSIZE = 0;
int KERNELSIZE = 0;

void convolve(double *matrix, double *kernel, double *target)
{
    /* 
     * Some required values. To be used to refer to indices 
     */
    int fh = KERNELSIZE/2, sh = KERNELSIZE - fh;
    double end;
    int offsetij = KERNELSIZE - 1;
    struct timespec t1, t2;

    /* 
     * Flip the kernel left-right and up-down.
     * Reduces the number of addition operations 
     * in indices.
     */
    double temp;
    for(int i = 0; i < fh; i ++)
        for(int j = 0; j < KERNELSIZE; j ++)
        {
            temp = kernel[i*KERNELSIZE + j];
            kernel[i*KERNELSIZE + j] = kernel[(KERNELSIZE - i - 1)*KERNELSIZE + j];
            kernel[(KERNELSIZE - i - 1)*KERNELSIZE + j] = temp;
        }
    for(int i = 0; i < KERNELSIZE; i ++)
        for(int j = 0; j < fh; j ++)
        {
            temp = kernel[i*KERNELSIZE + j];
            kernel[i*KERNELSIZE + j] = kernel[i*KERNELSIZE + KERNELSIZE - j - 1];
            kernel[i*KERNELSIZE + KERNELSIZE - j - 1] = temp;
        }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    /* 
     * Parallelize
     */
#pragma omp parallel for schedule(dynamic) \
    shared(matrix, kernel)
    for(int x = fh; x < MATSIZE-sh+1; x ++)
    {
        /*
         * The following assignment reduces KERNELSIZE*KERNELSIZE 
         * addition operations per element in matrix.
         */
        double *thismatrix = matrix + x - fh;

        for(int y = fh; y < MATSIZE-sh+1; y ++)
        {
            double sum = 0.0;
            for(int i = 0; i < KERNELSIZE; i ++)
            {
                /*
                 * offsety reduces KERNELSIZE more addition operations
                 */
                int offsety = i*MATSIZE + y - fh;

                for(int j = 0; j < KERNELSIZE; j ++)
                {
                    sum += thismatrix[offsety + j] * kernel[i*KERNELSIZE + j];
                }
            }
                            
            /*
             * Assign the result
             */
            target[x*MATSIZE + y] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t2);
    end = 1000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec)*1.0/1000000;
    cout << end << endl;

    return;
}


int main(int argc, char *argv[])
{
    if(argc < 3)
    {
        cout << "Please specify MATSIZE and KERNELSIZE. ./a.out <MATSIZE> <KERNELSIZE>" << endl;
        return 1;
    }
    MATSIZE = atoi(argv[1]);
    KERNELSIZE = atoi(argv[2]);
    double *mat = new double [MATSIZE*MATSIZE];
    double *kernel = new double [KERNELSIZE*KERNELSIZE];
    double *target = new double [MATSIZE*MATSIZE];
    int i, j;

    struct timespec t1, t2;
    double start, end;

    double sec;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = t1.tv_sec;
    srand(sec*sec);

    /* 
     * Assign random values
     */
    for(i = 0; i < MATSIZE*MATSIZE; i ++)
    {
        mat[i] = rand() % 10;
    }
    for(i = 0; i < KERNELSIZE*KERNELSIZE; i ++)
    {
        kernel[i] = rand()%10;
    }

    //memset(target, 0, sizeof(double)*MATSIZE*MATSIZE);
    
    convolve(mat, kernel, target);

    delete mat;
    delete target;
    delete kernel;
    return 0;
}
