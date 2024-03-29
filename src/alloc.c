#include <stdlib.h>
#include <stdio.h>

#include "mpi.h"

/* Allocate memory for a rows*cols array of floats.
 * The elements within a column are contiguous in memory, and columns
 * themselves are also contiguous in memory.
 */
float **alloc_floatmatrix(int cols, int rows)
{
    int i;
    float **m;
    if ((m = (float**) malloc(cols*sizeof(float*))) == NULL) {
        return NULL;
    }
    float *els = (float *) calloc(rows*cols, sizeof(float));
    if (els == NULL) {
        return NULL;
    } 
    for (i = 0; i < cols; i++) {
        m[i] = &els[rows * i];
    }
    return m;
} 

/* Allocate memory for a rows*cols array of chars. */
char **alloc_charmatrix(int cols, int rows)
{
    int i;
    char **m;
    if ((m = (char**) malloc(cols*sizeof(char*))) == NULL) {
        return NULL;
    }
    char *els = (char *) malloc(rows*cols*sizeof(char));
    if (els == NULL) {
        return NULL;
    } 
    for (i = 0; i < cols; i++) {
        m[i] = &els[rows * i];
    }
    return m;
} 

/* Free the memory of a matrix allocated with alloc_{float|char}matrix*/
void free_matrix(void *m)
{
    void **els = (void **) m;
    free(els[0]);   /* Deallocate the block of array elements */
    free(m);        /* Deallocate the block of column pointers */
}

/* Calcualates the number of blocks, and their respective widths
   Returns the start and stop index of block
 */
void partition(int nprocs, int proc, int imax,
    int *ileft, int *iright, int *neighbours)
{
    if (nprocs < 2) {
        *ileft = 0;
        *iright = imax;
        neighbours[0] = MPI_PROC_NULL;
        neighbours[1] = MPI_PROC_NULL;
        return;
    } else {
        int perproc = imax / nprocs;
        int diff = imax - (perproc * nprocs) + 2;

        fprintf(stderr, "Perproc is %d. Diff is %d. imax is %d\n", perproc, diff, imax);

        *ileft = (proc * perproc) + diff;
        *iright = *ileft + perproc;

        neighbours[0] = proc - 1;
        neighbours[1] = proc + 1;

        if (proc == 0) {
            *ileft = 0;
            neighbours[0] = MPI_PROC_NULL;
        }
        if (proc ==  nprocs - 1) {
            neighbours[1] = MPI_PROC_NULL;
        }
    }
}
