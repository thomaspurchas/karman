float **alloc_floatmatrix(int cols, int rows);
char **alloc_charmatrix(int cols, int rows);
void free_matrix(void *m);
void partition(int proc, int nprocs, int imax,
    int *ileft, int *iright);
