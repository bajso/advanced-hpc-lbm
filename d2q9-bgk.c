/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <mpi.h>

#define MASTER          0
#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"


/* MPI params */
int rank; // rank of a process
int nproc; // number of processes in the communicator
int flag; // check if MPI_Init() has been called
int tag = 0; // message tag
MPI_Status status; // struct used by MPI_Recv
int top_rank;
int bottom_rank;


/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

t_speed* sendbuf;
t_speed* recvbuf;


/*
** function prototypes
*/

/* split by rows, by the number of processes */
int calc_nrows_from_nproc(int rank, int nproc, int ny);

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int** total_obstacles_ptr, t_speed** total_cells_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, int** total_obstacles_ptr, t_speed** total_cells_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  t_speed* total_cells = NULL; /* total cells pointer for MPI Gather */
  int* total_obstacles = NULL; /* total obstacles pointer */


  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }


  /* initialise the MPI env */
  MPI_Init(&argc, &argv);

  /* check whether the initialisation was successful */
  MPI_Initialized(&flag);
  if (flag != 1) {
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  /*) number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  /* rank of the current process */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* determine process ranks above and below the current rank respecting periodic boundary conditions */
  top_rank = (rank == MASTER) ? (rank + nproc - 1) : (rank - 1);
  bottom_rank = (rank + 1) % nproc;


  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &total_obstacles, &total_cells);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);


  /* MPI type for t_speed */
  int blocklen[1] = {NSPEEDS};
  MPI_Datatype mpi_t_speed;
  MPI_Datatype type[1] = {MPI_FLOAT};
  MPI_Aint offset[1];
  offset[0] = offsetof(t_speed, speeds);
  MPI_Type_create_struct(1, blocklen, offset, type, &mpi_t_speed);
  MPI_Type_commit(&mpi_t_speed);

  int local_ny = calc_nrows_from_nproc(rank, nproc, params.ny);
  int segment_size = local_ny * params.nx; /* size of the segment that is scattered */
  int total_size = params.ny * params.nx;

  if (rank == MASTER) {
    printf("\nNproc: %d\nLocal ny: %d\nParams nx: %d\nParams ny: %d\nSegment: %d\nTotal: %d\n\n", nproc, local_ny, params.nx, params.ny, segment_size, total_size);
  }

  for (int tt = 0; tt < params.maxIters; tt++)
  {

    /* gather all subsets of obstacles at master */
    // TODO Why gather obstacles if no computation is done on them and they don't change

    MPI_Gather(obstacles, segment_size, MPI_INT, total_obstacles, segment_size, MPI_INT, MASTER, MPI_COMM_WORLD);

    printf("Rank %d CHECKPOINT 221\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);


    timestep(params, cells, tmp_cells, obstacles);


    // TODO this does not work
    /* gather all subsets of cells at master */
    /* gather and truncate the cells array without the halo exchanges */
    MPI_Gather(&cells[params.nx], total_size, mpi_t_speed, total_cells, total_size, mpi_t_speed, MASTER, MPI_COMM_WORLD);


    if (rank == MASTER)
    {
      /* better way is to have a local av_vels that and only gather everything once after the timestep is finished */
      // TODO maybe use MPI_Reduce()
      av_vels[tt] = av_velocity(params, cells, obstacles);
    }
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  if (rank == MASTER)
  {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, cells, obstacles, av_vels);
  }

  // MPI_Barrier(MPI_COMM_WORLD);
  /* need to finalise with every process to free up memory */
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, &total_obstacles, &total_cells);

  /* Finalise the MPI env */
  MPI_Finalize();

  return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  /* acc flow is done only on the topmost row, which is part of the last rank */
  if (rank == nproc - 1)
  {
    accelerate_flow(params, cells, obstacles);
    printf("Rank %d CHECKPOINT 283\n", rank);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  propagate(params, cells, tmp_cells);
  rebound(params, cells, tmp_cells, obstacles);
  collision(params, cells, tmp_cells, obstacles);
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

    /* modify the 2nd row of the grid */

    /* int jj = params.ny - 2 */
    /* 3 = -2 -1 for halo exchange row */
    int local_ny = calc_nrows_from_nproc(rank, nproc, params.ny);
    int jj = (local_ny + 2) - 3;

    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell is not occupied and
      ** we don't send a negative density */

      /* is it (jj + 1) even if int jj = -1 ? */
      /* or just obstacles (jj - 1) */

      if (!obstacles[ii + jj*params.nx]
          && (cells[ii + (jj + 1)*params.nx].speeds[3] - w1) > 0.f
          && (cells[ii + (jj + 1)*params.nx].speeds[6] - w2) > 0.f
          && (cells[ii + (jj + 1)*params.nx].speeds[7] - w2) > 0.f)
      {
        /* increase 'east-side' densities */
        cells[ii + (jj + 1)*params.nx].speeds[1] += w1;
        cells[ii + (jj + 1)*params.nx].speeds[5] += w2;
        cells[ii + (jj + 1)*params.nx].speeds[8] += w2;
        /* decrease 'west-side' densities */
        cells[ii + (jj + 1)*params.nx].speeds[3] -= w1;
        cells[ii + (jj + 1)*params.nx].speeds[6] -= w2;
        cells[ii + (jj + 1)*params.nx].speeds[7] -= w2;
      }
    }

  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells)
{

  /* MPI type for t_speed */
  int blocklen[1] = {NSPEEDS};
  MPI_Datatype mpi_t_speed;
  MPI_Datatype type[1] = {MPI_FLOAT};
  MPI_Aint offset[1];
  offset[0] = offsetof(t_speed, speeds);
  MPI_Type_create_struct(1, blocklen, offset, type, &mpi_t_speed);
  MPI_Type_commit(&mpi_t_speed);

  int local_ny = calc_nrows_from_nproc(rank, nproc, params.ny);
  int i;
  /* send up, receive from the bottom */

  // flatten 2D array
  // array[width * row + col] = value;

  // copy first Cells row into buffer
  for(i = 0; i < params.nx; i++) {
    sendbuf[i] = cells[(1 * params.nx) + i];
  }

  MPI_Sendrecv(sendbuf, params.nx, mpi_t_speed, top_rank, tag, recvbuf, params.nx, mpi_t_speed, bottom_rank, tag, MPI_COMM_WORLD, &status);

  // replace buffer elements with halo row in Cells
  for(i = 0; i < params.nx; i++) {
    // this is the bottom halo row - local_ny + 1
    cells[((local_ny + 1) * params.nx) + i] = recvbuf[i];
    tmp_cells[((local_ny + 1) * params.nx) + i] = recvbuf[i];
  }

  /* send down, receive from the top */

  // copy last Cells row into buffer
  for(i = 0; i < params.nx; i++) {
    sendbuf[i] = cells[(local_ny * params.nx) + i];
  }

  MPI_Sendrecv(sendbuf, params.nx, mpi_t_speed, bottom_rank, tag, recvbuf, params.nx, mpi_t_speed, top_rank, tag, MPI_COMM_WORLD, &status);

  // replace buffer elements with halo row in Cells
  for(i = 0; i < params.nx; i++) {
    // this is the top halo row - 0
    cells[(0 * params.nx) + i] = recvbuf[i];
    tmp_cells[(0 * params.nx) + i] = recvbuf[i];
  }

printf("Rank %d CHECKPOINT 383\n", rank);
MPI_Barrier(MPI_COMM_WORLD);

  /* loop over _all_ cells */
  for (int jj = 0; jj < local_ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = ((jj + 1) + 1) % local_ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = ((jj + 1) == 0) ? ((jj + 1) + local_ny - 1) : ((jj + 1) - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmp_cells[ii + (jj + 1)*params.nx].speeds[0] = cells[ii + (jj + 1)*params.nx].speeds[0]; /* central cell, no movement */
      tmp_cells[ii + (jj + 1)*params.nx].speeds[1] = cells[x_w + (jj + 1)*params.nx].speeds[1]; /* east */
      tmp_cells[ii + (jj + 1)*params.nx].speeds[2] = cells[ii + y_s*params.nx].speeds[2]; /* north */
      tmp_cells[ii + (jj + 1)*params.nx].speeds[3] = cells[x_e + (jj + 1)*params.nx].speeds[3]; /* west */
      tmp_cells[ii + (jj + 1)*params.nx].speeds[4] = cells[ii + y_n*params.nx].speeds[4]; /* south */
      tmp_cells[ii + (jj + 1)*params.nx].speeds[5] = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
      tmp_cells[ii + (jj + 1)*params.nx].speeds[6] = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      tmp_cells[ii + (jj + 1)*params.nx].speeds[7] = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
      tmp_cells[ii + (jj + 1)*params.nx].speeds[8] = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */
    }
  }

  printf("Rank %d CHECKPOINT 417\n", rank);
  MPI_Barrier(MPI_COMM_WORLD);

  return EXIT_SUCCESS;
}

int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  int local_ny = calc_nrows_from_nproc(rank, nproc, params.ny);

  /* loop over the cells in the grid */
  for (int jj = 0; jj < local_ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        cells[ii + (jj + 1)*params.nx].speeds[1] = tmp_cells[ii + (jj + 1)*params.nx].speeds[3];
        cells[ii + (jj + 1)*params.nx].speeds[2] = tmp_cells[ii + (jj + 1)*params.nx].speeds[4];
        cells[ii + (jj + 1)*params.nx].speeds[3] = tmp_cells[ii + (jj + 1)*params.nx].speeds[1];
        cells[ii + (jj + 1)*params.nx].speeds[4] = tmp_cells[ii + (jj + 1)*params.nx].speeds[2];
        cells[ii + (jj + 1)*params.nx].speeds[5] = tmp_cells[ii + (jj + 1)*params.nx].speeds[7];
        cells[ii + (jj + 1)*params.nx].speeds[6] = tmp_cells[ii + (jj + 1)*params.nx].speeds[8];
        cells[ii + (jj + 1)*params.nx].speeds[7] = tmp_cells[ii + (jj + 1)*params.nx].speeds[5];
        cells[ii + (jj + 1)*params.nx].speeds[8] = tmp_cells[ii + (jj + 1)*params.nx].speeds[6];
      }
    }
  }

  printf("Rank %d CHECKPOINT 449\n", rank);
  MPI_Barrier(MPI_COMM_WORLD);

  return EXIT_SUCCESS;
}

int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* don't consider occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells[ii + jj*params.nx].speeds[1]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[8]
                      - (tmp_cells[ii + jj*params.nx].speeds[3]
                         + tmp_cells[ii + jj*params.nx].speeds[6]
                         + tmp_cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells[ii + jj*params.nx].speeds[2]
                      + tmp_cells[ii + jj*params.nx].speeds[5]
                      + tmp_cells[ii + jj*params.nx].speeds[6]
                      - (tmp_cells[ii + jj*params.nx].speeds[4]
                         + tmp_cells[ii + jj*params.nx].speeds[7]
                         + tmp_cells[ii + jj*params.nx].speeds[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[ii + jj*params.nx].speeds[kk] = tmp_cells[ii + jj*params.nx].speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_cells[ii + jj*params.nx].speeds[kk]);
        }
      }
    }
  }

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int** total_obstacles_ptr, t_speed** total_cells_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */

  int local_ny = calc_nrows_from_nproc(rank, nproc, params->ny);

  /* split rows, +1 top +1 bottom for halo exchange */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * ((local_ny + 2) * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* allocate memory for total cells array */
  *total_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*total_cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * ((local_ny + 2) * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */

  /* no need for + 2 since no calculation is performed there */
  /* every process allocates memory for this array */
  *obstacles_ptr = malloc(sizeof(int) * (local_ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise for MPI Scatter */
  *total_obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*total_obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* allocate buffer space */
  sendbuf = (t_speed*)malloc(sizeof(t_speed) * params->nx); // fits one column of cells
  recvbuf = (t_speed*)malloc(sizeof(t_speed) * params->nx);


  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

 for (int jj = 0; jj < local_ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* first and last rows are for halo exchange - 0|1|1|1|1|0 */
      /* add + 1 to jj access the second row */

      /* centre */
      (*cells_ptr)[ii + (jj + 1)*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + (jj + 1)*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + (jj + 1)*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + (jj + 1)*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + (jj + 1)*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + (jj + 1)*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + (jj + 1)*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + (jj + 1)*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + (jj + 1)*params->nx].speeds[8] = w2;


      /* first set all cells in obstacle array to zero */
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }
  /* does it even need to be initialised? */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*total_obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  int segment_size = local_ny * params->nx;
  if (rank == MASTER)
  {
    // printf("\nSegement size:%d\n", segment_size);

    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");

    if (fp == NULL)
    {
      sprintf(message, "could not open input obstacles file: %s", obstaclefile);
      die(message, __LINE__, __FILE__);
    }

    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
    {
      /* some checks */
      if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

      if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

      if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

      if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

      /* assign to array */
      (*total_obstacles_ptr)[xx + yy*params->nx] = blocked;
    }

    /* and close the file */
    fclose(fp);
  }

  /* scatter the obstacles array to all other processes */
  /* total_obstacles_ptr, obstacles_ptr determine the size of the buffer */

  /* size of the segment that is scattered */

  /* TODO buffer size = 1 or segment size?*/

  MPI_Scatter(*total_obstacles_ptr, segment_size, MPI_INT, *obstacles_ptr, segment_size, MPI_INT, MASTER, MPI_COMM_WORLD);

  // printf("\nRank %d\nValue total_obstacles_ptr %d\n", rank, **total_obstacles_ptr );
  // printf("Rank %d\nValue obstacles_ptr %d\n", rank, **obstacles_ptr );

  printf("Rank %d CHECKPOINT 727\n", rank);
  MPI_Barrier(MPI_COMM_WORLD);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */

  /* this could also be only performed by a master process */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, int** total_obstacles_ptr, t_speed** total_cells_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  free(*total_obstacles_ptr);
  *total_obstacles_ptr = NULL;

  free(*total_cells_ptr);
  *total_cells_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

int calc_nrows_from_nproc(int rank, int nproc, int ny)
{
  int local_ny;

  local_ny = ny / nproc;
  if ((ny % nproc) != 0) {  /* if there is a remainder */
    if (rank == nproc - 1) {
      local_ny += ny % nproc;  /* add remainder to last rank */
    }
  }

  return local_ny;
}
