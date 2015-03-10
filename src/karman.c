#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>

#include "mpi.h"

#include "alloc.h"
#include "boundary.h"
#include "datadef.h"
#include "init.h"
#include "simulation.h"

void write_bin(float **u, float **v, float **p, char **flag,
     int gimax, int gjmax, float xlength, float ylength, char *file);

int read_bin(float **u, float **v, float **p, char **flag,
    int gimax, int gjmax, float xlength, float ylength, char *file,
    int ileft, int iright);

void send_column(void *column, MPI_Datatype type, int jmax,
    int id, int to, MPI_Request *request);
void receive_column(void *column, MPI_Datatype type, int jmax, int id, int from);

void send_boundaries(float **u, float **v, int imax, int jmax, MPI_Request requests[]);
void receive_boundaries(float **u, float **v, int imax, int jmax);

static void print_usage(void);
static void print_version(void);
static void print_help(void);

static char *progname;

int verbose = 1;          /* Verbosity level */

int proc = 0;                       /* Rank of the current process */
int nprocs = 0;                /* Number of processes in communicator */

int ileft, iright;           /* Array bounds for each processor */
int neighbours[2] = {
    MPI_PROC_NULL, MPI_PROC_NULL}; /* Array storage of process neighbours */

#define PACKAGE "karman"
#define VERSION "1.0"

/* Command line options */
static struct option long_opts[] = {
    { "del-t",   1, NULL, 'd' },
    { "help",    0, NULL, 'h' },
    { "imax",    1, NULL, 'x' },
    { "infile",  1, NULL, 'i' },
    { "jmax",    1, NULL, 'y' },
    { "outfile", 1, NULL, 'o' },
    { "t-end",   1, NULL, 't' },
    { "verbose", 1, NULL, 'v' },
    { "version", 1, NULL, 'V' },
    { 0,         0, 0,    0   } 
};
#define GETOPTS "d:hi:o:t:v:Vx:y:"

int main(int argc, char *argv[])
{
    float gxlength = 22.0;     /* Width of global simulated domain */
    float gylength = 4.1;      /* Height of global simulated domain */
    float xlength = gxlength;  /* Width of local simulated domain */
    float ylength = gylength;  /* Height of local simulated domain */
    int gimax = 660;          /* Number of global cells horizontally */
    int gjmax = 120;          /* Number of global cells vertically */
    int imax = gimax;         /* Number of local cells horizontally */
    int jmax = gjmax;         /* Number of local cells vertically */


    char *infile;             /* Input raw initial conditions */
    char *outfile;            /* Output raw simulation results */

    float t_end = 2.1;        /* Simulation runtime */
    float del_t = 0.003;      /* Duration of each timestep */
    float tau = 0.5;          /* Safety factor for timestep control */

    int itermax = 100;        /* Maximum number of iterations in SOR */
    float eps = 0.001;        /* Stopping error threshold for SOR */
    float omega = 1.7;        /* Relaxation parameter for SOR */
    float gamma = 0.9;        /* Upwind differencing factor in PDE
                                 discretisation */

    float Re = 150.0;         /* Reynolds number */
    float ui = 1.0;           /* Initial X velocity */
    float vi = 0.0;           /* Initial Y velocity */

    float t, delx, dely;
    int  i, j, itersor = 0, ifluid = 0, ibound = 0;
    float res;
    float **u, **v, **p, **rhs, **f, **g;
    char  **flag;
    int init_case, iters = 0;
    int show_help = 0, show_usage = 0, show_version = 0;

    MPI_Request requests[4];
    MPI_Status statuses[4];


    /* Before doing any argument processing, let MPI do it's thing */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);

    progname = argv[0];
    infile = strdup("karman.bin");
    outfile = strdup("karman.bin");

    int optc;
    while ((optc = getopt_long(argc, argv, GETOPTS, long_opts, NULL)) != -1) {
        switch (optc) {
            case 'h':
                show_help = 1;
                break;
            case 'V':
                show_version = 1;
                break;
            case 'v':
                verbose = atoi(optarg);
                break;
            case 'x':
                gimax = atoi(optarg);
                imax = gimax;
                break;
            case 'y':
                gjmax = atoi(optarg);
                jmax = gjmax;
                break;
            case 'i':
                free(infile);
                infile = strdup(optarg);
                break;
            case 'o':
                free(outfile);
                outfile = strdup(optarg);
                break;
            case 'd':
                del_t = atof(optarg);
                break;
            case 't':
                t_end = atof(optarg);
                break;
            default:
                show_usage = 1;
        }
    }
    if (show_usage || optind < argc) {
        print_usage();
        return 1;
    }
    
    if (show_version) {
        print_version();
        if (!show_help) {
            return 0;
        }
    }
    
    if (show_help) {
        print_help();
        return 0;
    }

    /* Calculate block partitions */
    partition(nprocs, proc, imax, &ileft, &iright, neighbours);

    /* Calculate new imax and xlength for this process */
    imax = iright - ileft;
    xlength = (gxlength/gimax) * imax;
    delx = xlength/imax;
    dely = ylength/jmax;

    if (proc == 0) {
        fprintf(stderr, "Number of processes = %d\n", nprocs);
    }
    if (verbose > 1) {
        fprintf(stderr, "ileft is %d. iright is %d. imax is %d\n", ileft, iright, imax);
    }

    /* Allocate arrays */
    v    = alloc_floatmatrix(imax+2, jmax+2);
    u    = alloc_floatmatrix(imax+2, jmax+2);
    f    = alloc_floatmatrix(imax+2, jmax+2);
    g    = alloc_floatmatrix(imax+2, jmax+2);
    p    = alloc_floatmatrix(imax+2, jmax+2);
    rhs  = alloc_floatmatrix(imax+2, jmax+2); 
    flag = alloc_charmatrix(imax+2, jmax+2);                    

    if (!u || !v || !f || !g || !p || !rhs || !flag) {
        fprintf(stderr, "Couldn't allocate memory for matrices.\n");
        return 1;
    }

    /* Read in initial values from a file if it exists */
    init_case = read_bin(u, v, p, flag, gimax, gjmax, gxlength, gylength, infile, ileft, iright);

    send_boundaries(u, v, imax+2, jmax+2, requests);
    receive_boundaries(u, v, imax+2, jmax+2);

    MPI_Waitall(2, requests, statuses);

    if (init_case > 0) {
        /* Error while reading file */
        MPI_Finalize();
        return 1;
    }

    if (init_case < 0) {
        /* Check that we are in single process mode */
        if (nprocs > 1) {
            fprintf(stderr, "This version of karman does not support\n");
            fprintf(stderr,  "generating init conditions in multiprocess mode\n\n");
            fprintf(stderr,  "Please run with a single process and a timestep of 0\n");
            fprintf(stderr,  "to generate init conditions bin\n");

            MPI_Finalize();
            return 1;
        } else {
            /* Set initial values if file doesn't exist */
            for (i=0;i<=imax+1;i++) {
                for (j=0;j<=jmax+1;j++) {
                    u[i][j] = ui;
                    v[i][j] = vi;
                    p[i][j] = 0.0;
                }
            }
            init_flag(flag, imax, jmax, delx, dely, &ibound);
            apply_boundary_conditions(u, v, flag, imax, jmax, ui, vi);
        }
    }

    /* Main loop */
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        set_timestep_interval(&del_t, imax, jmax, delx, dely, u, v, Re, tau);
        // fprintf(stderr, "%d: Calculated timestep at %f\n", proc, del_t);
        /* Each process will have calculated a different time step
           now pick the lowest timestep, and use it in every process
        */
        MPI_Allreduce(MPI_IN_PLACE, &del_t, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        if (proc == 0 && verbose > 1) {
            fprintf(stderr, "Time step is: %f\n", del_t);
        }

        ifluid = (imax * jmax) - ibound;

        compute_tentative_velocity(u, v, f, g, flag, imax, jmax,
            del_t, delx, dely, gamma, Re);

        compute_rhs(f, g, rhs, flag, imax, jmax, del_t, delx, dely);

        if (ifluid > 0) {
            itersor = poisson(p, rhs, flag, imax, jmax, delx, dely,
                        eps, itermax, omega, &res, ifluid);
        } else {
            itersor = 0;
        }

        if (proc == 0 && verbose > 1) {
            printf("%d t:%g, del_t:%g, SOR iters:%3d, res:%e, bcells:%d\n",
                iters, t+del_t, del_t, itersor, res, ibound);
        }

        update_velocity(u, v, f, g, p, flag, imax, jmax, del_t, delx, dely);

        send_boundaries(u, v, imax+2, jmax+2, requests);
        receive_boundaries(u, v, imax+2, jmax+2);

        MPI_Waitall(2, requests, statuses);

        apply_boundary_conditions(u, v, flag, imax, jmax, ui, vi);
        // fprintf(stderr, "%d: Starting next loop\n", proc);
    } /* End of main loop */

    fprintf(stderr, "%d: Completed main loop\n", proc);
  
    if (outfile != NULL && strcmp(outfile, "") != 0) {
        write_bin(u, v, p, flag, gimax, gjmax, gxlength, gylength, outfile);
    }

    free_matrix(u);
    free_matrix(v);
    free_matrix(f);
    free_matrix(g);
    free_matrix(p);
    free_matrix(rhs);
    free_matrix(flag);

    /* Before we exit, let MPI finish */
    MPI_Finalize();

    return 0;
}

/* Save the simulation state to a file */
void write_bin(float **u, float **v, float **p, char **flag,
    int gimax, int gjmax, float xlength, float ylength, char* file)
{
    int i, j;
    if (proc == 0) {
        FILE *fp;

        fp = fopen(file, "wb"); 

        if (fp == NULL) {
            fprintf(stderr, "Could not open file '%s': %s\n", file,
                strerror(errno));
            return;
        }

        fwrite(&gimax, sizeof(int), 1, fp);
        fwrite(&gjmax, sizeof(int), 1, fp);
        fwrite(&xlength, sizeof(float), 1, fp);
        fwrite(&ylength, sizeof(float), 1, fp);

        int perproc = gimax / nprocs;
        int diff = (gimax - (perproc * nprocs)) + 2;
        int thisproc = 0;

        for (i=0; i<nprocs; i++) {
            if (i == 0) {
                thisproc = perproc + diff;
            } else {
                thisproc = perproc;
            }
            for (j=0;j<thisproc;j++) {
                if (i > 0) {
                    if (verbose > 1) {
                        fprintf(stderr, "%d: receving column %d\n", proc, i);
                    }
                    receive_column(u[j], MPI_FLOAT, gjmax+2, j+1, i);
                    receive_column(v[j], MPI_FLOAT, gjmax+2, j+1, i);
                    receive_column(p[j], MPI_FLOAT, gjmax+2, j+1, i);
                    receive_column(flag[j], MPI_CHAR, gjmax+2, j+1, i);
                    if (verbose > 1) {
                        fprintf(stderr, "%d: receved column %d\n", proc, i);
                    }
                }
                fwrite(u[j], sizeof(float), gjmax+2, fp);
                fwrite(v[j], sizeof(float), gjmax+2, fp);
                fwrite(p[j], sizeof(float), gjmax+2, fp);
                fwrite(flag[j], sizeof(char), gjmax+2, fp);
            }
            if (verbose > 1) {
                fprintf(stderr, "Got data from %d\n", i);
            }
        }
        fclose(fp);

    } else {
        int imax = iright - ileft;

        MPI_Request requests[4];
        MPI_Status statuses[4];

        for (i=1; i<imax+1; i++) {
            if (verbose > 1) {
                fprintf(stderr, "%d: sending column %d\n", proc, i);
            }
            send_column(u[i], MPI_FLOAT, gjmax+2, i, 0, &requests[0]);
            send_column(v[i], MPI_FLOAT, gjmax+2, i, 0, &requests[1]);
            send_column(p[i], MPI_FLOAT, gjmax+2, i, 0, &requests[2]);
            send_column(flag[i], MPI_CHAR, gjmax+2, i, 0, &requests[3]);
            
            MPI_Waitall(4, requests, statuses);

            if (verbose > 1) {
                fprintf(stderr, "%d: sent column %d\n", proc, i);
            }
        }

        fprintf(stderr, "%d: Finished sending data\n", proc);

        return 0;
    }
}

/* Read the simulation state from a file */
int read_bin(float **u, float **v, float **p, char **flag,
    int gimax, int gjmax, float xlength, float ylength, char* file,
    int ileft, int iright)
{
    int i,j;

    if (proc == 0) {
        FILE *fp;

        if (file == NULL) return -1;

        if ((fp = fopen(file, "rb")) == NULL) {
            fprintf(stderr, "Could not open file '%s': %s\n", file,
                strerror(errno));
            fprintf(stderr, "Generating default state instead.\n");
            return -1;
        }

        fread(&i, sizeof(int), 1, fp);
        fread(&j, sizeof(int), 1, fp);
        float xl, yl;
        fread(&xl, sizeof(float), 1, fp);
        fread(&yl, sizeof(float), 1, fp);

        if (i!=gimax || j!=gjmax) {
            fprintf(stderr, "Warning: gimax/gjmax have wrong values in %s\n", file);
            fprintf(stderr, "%s's gimax = %d, gjmax = %d\n", file, i, j);
            fprintf(stderr, "Program's gimax = %d, gjmax = %d\n", gimax, gjmax);
            return 1;
        }
        if (xl!=xlength || yl!=ylength) {
            fprintf(stderr, "Warning: xlength/ylength have wrong values in %s\n", file);
            fprintf(stderr, "%s's xlength = %g,  ylength = %g\n", file, xl, yl);
            fprintf(stderr, "Program's xlength = %g, ylength = %g\n", xlength,
                ylength);
            return 1;
        }

        int perproc = gimax / nprocs;
        int diff = gimax - (perproc * nprocs) + 2;
        int offset, byteoffset = 0;

        MPI_Request requests[4];
        MPI_Status statuses[4];

        /* Calculate the offset in bytes for proc = 1 */
        byteoffset = diff * (sizeof(float) * (gjmax+2) * 3);
        byteoffset += diff * (sizeof(char) * (gjmax+2) * 1);
        byteoffset += perproc * (sizeof(float) * (gjmax+2) * 3);
        byteoffset += perproc * (sizeof(char) * (gjmax+2) * 1);
        /* Seek to offset */
        offset = ftell(fp);
        fseek(fp, byteoffset, SEEK_CUR);
        // fprintf(stderr, "Offset is %d\n", ftell(fp));
        // fprintf(stderr, "Perproc is %d\n", perproc);
        /* Read in data for each process, starting at proc 1 */
        for (i=1; i<nprocs; i++) {
            // fprintf(stderr, "Read from %l to",  ftell(fp));
            for (j=1; j<perproc+1; j++) {
                fread(u[j], sizeof(float), gjmax+2, fp);
                send_column(u[j], MPI_FLOAT, gjmax+2, j, i, &requests[0]);

                fread(v[j], sizeof(float), gjmax+2, fp);
                send_column(v[j], MPI_FLOAT, gjmax+2, j, i, &requests[1]);

                fread(p[j], sizeof(float), gjmax+2, fp);
                send_column(p[j], MPI_FLOAT, gjmax+2, j, i, &requests [2]);

                fread(flag[j], sizeof(char), gjmax+2, fp);
                send_column(flag[j], MPI_CHAR, gjmax+2, j, i, &requests[3]);

                MPI_Waitall(4, requests, statuses);
            }
            // fprintf(stderr, "File location: %d\n", ftell(fp));
        }
        /* Read in data for proc 0 now */
        fseek(fp, offset, SEEK_SET);
        for (j=0; j<perproc + diff; j++) {
            fread(u[j], sizeof(float), gjmax+2, fp);
            fread(v[j], sizeof(float), gjmax+2, fp);
            fread(p[j], sizeof(float), gjmax+2, fp);
            fread(flag[j], sizeof(char), gjmax+2, fp);
        }
        // fprintf(stderr, "File location: %d\n", ftell(fp));

        fclose(fp);
        // fprintf(stderr, "Proc %d has imax %d\n", proc, perproc + diff);
        return 0;
    } else {
        int imax = iright - ileft;

        for (i=1; i<imax+1; i++) {
            receive_column(u[i], MPI_FLOAT, gjmax+2, i, 0);
            receive_column(v[i], MPI_FLOAT, gjmax+2, i, 0);
            receive_column(p[i], MPI_FLOAT, gjmax+2, i, 0);
            receive_column(flag[i], MPI_CHAR, gjmax+2, i, 0);
            if (verbose > 1) {
                fprintf(stderr, "Proc %d got column %d\n", proc, i);
            }
        }
        if (verbose > 1) {
            fprintf(stderr, "Proc %d has data\n", proc);
        }

        return 0;
    }  
}

void send_column(void *column, MPI_Datatype type, int jmax,
    int id, int to, MPI_Request *request)
{
    MPI_Isend(column, jmax, type, to, id, MPI_COMM_WORLD, request);
}

void receive_column(void *column, MPI_Datatype type, int jmax, int id, int from)
{
    MPI_Status status;
    MPI_Recv(column, jmax, type, from, id, MPI_COMM_WORLD, &status);
}

void send_boundaries(float **u, float **v, int imax, int jmax, MPI_Request *requests)
{
    requests[0] = MPI_REQUEST_NULL;
    requests[1] = MPI_REQUEST_NULL;
    requests[2] = MPI_REQUEST_NULL;
    requests[3] = MPI_REQUEST_NULL;

    if (neighbours[0] != MPI_PROC_NULL) {
        send_column(u[1], MPI_FLOAT, jmax, 0, neighbours[0], &requests[0]);
        send_column(v[1], MPI_FLOAT, jmax, 1, neighbours[0], &requests[1]);
    }

    if (neighbours[1] != MPI_PROC_NULL) {
        send_column(u[imax-2], MPI_FLOAT, jmax, 0, neighbours[1], &requests[2]);
        send_column(v[imax-2], MPI_FLOAT, jmax, 1, neighbours[1], &requests[3]);
    }
    if (verbose > 1) {
        (stderr, "%d: Sent all boundries \n", proc); 
    }

}

void receive_boundaries(float **u, float **v, int imax, int jmax)
{
    if (neighbours[0]) {
        receive_column(u[0], MPI_FLOAT, jmax, 0, neighbours[0]);
        receive_column(v[0], MPI_FLOAT, jmax, 1, neighbours[0]);
    }

    if (neighbours[1]) {
        receive_column(u[imax-1], MPI_FLOAT, jmax, 0, neighbours[1]);
        receive_column(v[imax-1], MPI_FLOAT, jmax, 1, neighbours[1]);
    }
    if (verbose > 1){
        fprintf(stderr, "%d: Got all boundries \n", proc);
    }
}

static void print_usage(void)
{
    fprintf(stderr, "Try '%s --help' for more information.\n", progname);
}

static void print_version(void)
{
    fprintf(stderr, "%s %s\n", PACKAGE, VERSION);
}

static void print_help(void)
{
    fprintf(stderr, "%s. A simple computational fluid dynamics tutorial.\n\n",
        PACKAGE);
    fprintf(stderr, "Usage: %s [OPTIONS]...\n\n", progname);
    fprintf(stderr, "  -h, --help            Print a summary of the options\n");
    fprintf(stderr, "  -V, --version         Print the version number\n");
    fprintf(stderr, "  -v, --verbose=LEVEL   Set the verbosity level. 0 is silent\n");
    fprintf(stderr, "  -x, --imax=IMAX       Set the number of interior cells in the X direction\n");
    fprintf(stderr, "  -y, --jmax=JMAX       Set the number of interior cells in the Y direction\n");
    fprintf(stderr, "  -t, --t-end=TEND      Set the simulation end time\n");
    fprintf(stderr, "  -d, --del-t=DELT      Set the simulation timestep size\n");
    fprintf(stderr, "  -i, --infile=FILE     Read the initial simulation state from this file\n");
    fprintf(stderr, "                        (default is 'karman.bin')\n");
    fprintf(stderr, "  -o, --outfile=FILE    Write the final simulation state to this file\n");
    fprintf(stderr, "                        (default is 'karman.bin')\n");
}
