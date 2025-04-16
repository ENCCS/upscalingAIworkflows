.. _train_contain:

Training Neural Networks using Containers
======================================================

We discussed already different methods of scaling
for the training of the network. The essential part of any scaling
scheme is the communication among the processors whether it is 
a bunch of CPUs or GPUs. For the communication between CPUs, 
`MPI (Message Passing Interface) <https://en.wikipedia.org/wiki/
Message_Passing_Interface>`_ is a widely used standard. 
MPI is a well-established standard and it is used for
exchanging messages/data between processes in a parallel application.
If you've been involved in developing or working with computational
science software, you may already be familiar with MPI and running MPI
applications.

As for the communication between GPUs, depending on vendor providing GPUs, 
there are library, similar to MPI. GPUs which are available on Vega cluster 
are NVIDIA GPUs. The standard for communication for such GPUs is the NVIDIA 
Collective Communication Library `(NCCL) <https://developer.nvidia.com/nccl>`_ 
(NCCL, pronounced “Nickel”), 
partly as discussed in :doc:`tf_mltgpus`. Nvidia introduces NCCL as a library 
that enables multi-GPU and multi-node communication primitives optimized 
for NVIDIA GPUs and Networking that are topology-aware and can be easily integrated 
into applications. 
NCCL implements both collective communication and point-to-point send/receive 
primitives. It is not a full-blown parallel programming framework; rather, it is 
a library focused on accelerating inter-GPU communication.

NCCL provides the following collective communication primitives :

- AllReduce
- Broadcast
- Reduce
- AllGather
- ReduceScatter
Additionally, it allows for point-to-point send/receive communication which allows 
for scatter, gather, or all-to-all operations (`NCCL doc <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html>`_).

In this section of the workshop, we will see how these two libraries will be in
use during the training of a network using containers.

MPI codes with Singularity containers
-------------------------------------

We've already seen that building Singularity containers can be
impractical without root access. While it is unlikely to have
root access on a large institutional, regional or national cluster,
building a container directly on the target platform is not normally
an option, the Vega staff cluster has generously given us the necessary
privileges for creating containers.

One of the reasons we mentioned for using containers is their portability across 
different platforms/machines. However, it is not the case when we need to
create containers for training a network on specific cluster. If our target platform 
uses `OpenMPI <https://www.open-mpi.org/>`_,
one of the two widely used source MPI implementations, we can
build/install a compatible OpenMPI version on our local build
platform, or directly within the image as part of the image build
process. We can then build our code that requires MPI, either
interactively in an image sandbox or via a definition file.

While building a container on a local system that is intended for use
on a remote HPC platform does provide some level of portability, if
you're after the best possible performance, it can present some
issues. The version of MPI in the container will need to be built and
configured to support the hardware on your target platform if the best
possible performance is to be achieved. Where a platform has
specialist hardware with proprietary drivers, building on a different
platform with different hardware present means that building with the
right driver support for optimal performance is not likely to be
possible. This is especially true if the version of MPI available is
different (but compatible). Singularity's `MPI documentation
<https://sylabs.io/guides/3.9/user-guide/mpi.html>`_ highlights two
different models for working with MPI codes namely, the hybrid and bind methods.

The basic idea behind the Hybrid Approach is when you execute a 
Singularity container with MPI code, you will call ``mpiexec`` 
or a similar launcher on the ``singularity`` command itself. 
The MPI process outside of the container will then work in tandem with MPI 
inside the container and the containerized MPI code to instantiate the job.
Similarly, the basic idea behind the Bind Approach is 
to start the MPI application by calling the MPI launcher (e.g., ``mpirun``) 
from the host. The main difference between the hybrid and bind approach is 
the fact that with the bind approach, the container usually does not include 
any MPI implementation. This means that SingularityCE needs to mount/bind the 
MPI available on the host into the container.

The `hybrid model
<https://sylabs.io/guides/3.9/user-guide/mpi.html#hybrid-model>`_ that
we'll be looking at here involves using the MPI executable from the
MPI installation on the host system to launch singularity and run the
application within the container.  The application in the container is
linked against and uses the MPI installation within the container
which, in turn, communicates with the MPI daemon process running on
the host system. In the following sections we'll look at building a
Singularity image containing a small MPI application that can then be
run using the hybrid model.

The simplest MPI example
------------------------
 
Let's start with the simplest example of running an app within a container. This 
example will show the backbone of scaling an app using MPI primitives.
 
Create a new directory and save the ``mpitest.c`` given below.
 
..  code-block :: c++
 
   #include <mpi.h>
   #include <stdio.h>
   #include <stdlib.h>
 
   int main (int argc, char **argv) {
         int rc;
         int size;
         int myrank;
 
         rc = MPI_Init (&argc, &argv);
         if (rc != MPI_SUCCESS) {
                 fprintf (stderr, "MPI_Init() failed");
                 return EXIT_FAILURE;
         }
 
         rc = MPI_Comm_size (MPI_COMM_WORLD, &size);
         if (rc != MPI_SUCCESS) {
                 fprintf (stderr, "MPI_Comm_size() failed");
                 goto exit_with_error;
         }
 
         rc = MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
         if (rc != MPI_SUCCESS) {
                 fprintf (stderr, "MPI_Comm_rank() failed");
                 goto exit_with_error;
         }
 
         fprintf (stdout, "Hello, I am rank %d/%d\n", myrank, size);
 
         MPI_Finalize();
 
         return EXIT_SUCCESS;
   }

A possible def file for the app above is given below. 

.. code-block :: docker

   Bootstrap: docker
   From: ubuntu:18.04
 
   %files
     mpitest.c /opt
 
   %environment
     # Point to OMPI binaries, libraries, man pages
     export OMPI_DIR=/opt/ompi
     export PATH="$OMPI_DIR/bin:$PATH"
     export LD_LIBRARY_PATH="$OMPI_DIR/lib:$LD_LIBRARY_PATH"
     export MANPATH="$OMPI_DIR/share/man:$MANPATH"
 
   %post
     echo "Installing required packages..."
     apt-get update && apt-get install -y wget git bash gcc gfortran g++ make file
 
     echo "Installing Open MPI"
     export OMPI_DIR=/opt/ompi
     export OMPI_VERSION=4.0.5
     export OMPI_URL="https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-$OMPI_VERSION.tar.bz2"
     mkdir -p /tmp/ompi
     mkdir -p /opt
     # Download
     cd /tmp/ompi && wget -O openmpi-$OMPI_VERSION.tar.bz2 $OMPI_URL && tar -xjf openmpi-$OMPI_VERSION.tar.bz2
     # Compile and install
     cd /tmp/ompi/openmpi-$OMPI_VERSION && ./configure --prefix=$OMPI_DIR && make -j8 install
 
     # Set env variables so we can compile our application
     export PATH=$OMPI_DIR/bin:$PATH
     export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH
 
     echo "Compiling the MPI application..."
     cd /opt && mpicc -o mpitest mpitest.c


A quick recap of what the above definition file is doing:

 - The image is being bootstrapped from the ``ubuntu:18.04`` Docker
   image.
 - In the ``%environment`` section: Set an environment variable that
   will be available within all containers run from the generated
   image.
 - In the ``%post`` section:

   - Ubuntu's ``apt-get`` package manager is used to update the package
     directory and then install the compilers and other libraries
     required for the OpenMPI build.
   - The OpenMPI ``.tar.gz`` file is extracted and the configure, build and
     install steps are run.

We have the option of either compiling ``mpitest.c`` directly on the cluster, 
or compiling it inside the container. For learning purposes, let's compile the 
code inside the container. 

To create the container we use 

.. code-block:: bash

   singularity build --fakeroot --sandbox mpi_hybrid mpi_hybrid.def
   singularity build mpi_hybrid.sif mpi_hybrid


And to run the code on ``8`` processors we should use 
the command

.. code-block :: bash

   mpirun -n 8 singularity exec mpi_hybrid.sif /opt/mpitest

The output should look like 

.. code-block :: bash

   Hello, I am rank 1/8
   Hello, I am rank 2/8
   Hello, I am rank 3/8
   Hello, I am rank 4/8
   Hello, I am rank 5/8
   Hello, I am rank 6/8
   Hello, I am rank 7/8
   Hello, I am rank 0/8

Let's analyze what just happened. The ``mpitest`` app sent a ``Hello, I am rank X/Y``
message from within the container sent across ``8`` processors. For this process to
happen, ``mpirun`` runs a copy of the ``mpi_hybrid.sif`` container across the ``8`` 
processors and execute ``/opt/mpitest`` inside the container as we asked. 

MPI Ping-Pong 
-------------

The above example, did not have communicating between CPUs. To have a full-fledged
MPI app that can scale with number of CPUs, communication is a must. Let's take a
look at how communication works using MPI within a container. To that end, we will
use what is a common test for MPI communication. `Pingpong test` is a routine during 
which a message is sent and received in pingpong fashion between two processor.
As result of such test, the latency and bandwidth can be calculated.

One can either use the ``pingpong`` method as given in `HLRS MPI course 
<https://www.hlrs.de/about-us/media-publications/teaching-training-material/>`_ 
below

.. code-block:: fortran

   PROGRAM pingpong

   !==============================================================!
   !                                                              !
   ! This file has been written as a sample solution to an        !
   ! exercise in a course given at the High Performance           !
   ! Computing Centre Stuttgart (HLRS).                           !
   ! The examples are based on the examples in the MPI course of  !
   ! the Edinburgh Parallel Computing Centre (EPCC).              !
   ! It is made freely available with the understanding that      !
   ! every copy of this file must include this header and that    !
   ! HLRS and EPCC take no responsibility for the use of the      !
   ! enclosed teaching material.                                  !
   !                                                              !
   ! Authors: Joel Malard, Alan Simpson,            (EPCC)        !
   !          Rolf Rabenseifner, Traugott Streicher (HLRS)        !
   !                                                              !
   ! Contact: rabenseifner@hlrs.de                                !
   !                                                              !
   ! Purpose: A program to try MPI_Ssend and MPI_Recv.            !
   !                                                              !
   ! Contents: F-Source                                           !
   !                                                              !
   !==============================================================!
   
     USE mpi
   
     IMPLICIT NONE
   
     INTEGER proc_a
     PARAMETER(proc_a=0)
               
     INTEGER proc_b
     PARAMETER(proc_b=1)                
   
     INTEGER ping
     PARAMETER(ping=17)
           
     INTEGER pong
     PARAMETER(pong=23)        
   
     INTEGER number_of_messages 
     PARAMETER (number_of_messages=50)
   
     INTEGER start_length 
     PARAMETER (start_length=8)
   
     INTEGER length_factor 
     PARAMETER (length_factor=64)
   
     INTEGER max_length                ! 2 Mega 
     PARAMETER (max_length=2097152)
   
     INTEGER number_package_sizes 
     PARAMETER (number_package_sizes=8)
   
     INTEGER i, j
     INTEGER(KIND=MPI_ADDRESS_KIND) lb, size_of_real
   
     INTEGER length
    
     DOUBLE PRECISION start, finish, time, transfer_time
     INTEGER status(MPI_STATUS_SIZE)
      
     REAL buffer(max_length)
   
     INTEGER ierror, my_rank, size
   
   
     CALL MPI_INIT(ierror)
   
     CALL MPI_COMM_RANK(MPI_COMM_WORLD, my_rank, ierror)
     CALL MPI_TYPE_GET_EXTENT(MPI_REAL, lb, size_of_real, ierror) 
   
     IF (my_rank .EQ. proc_a) THEN
        WRITE (*,*) "message size   transfertime    bandwidth"
     END IF
   
     length = start_length
   
     DO j = 1, number_package_sizes
   
        IF (my_rank .EQ. proc_a) THEN
              CALL MPI_SEND(buffer, length, MPI_REAL, proc_b, ping, MPI_COMM_WORLD, ierror)
              CALL MPI_RECV(buffer, length, MPI_REAL, proc_b, pong, MPI_COMM_WORLD, status, ierror)
        ELSE IF (my_rank .EQ. proc_b) THEN
              CALL MPI_RECV(buffer, length, MPI_REAL, proc_a, ping, MPI_COMM_WORLD, status, ierror)
              CALL MPI_SEND(buffer, length, MPI_REAL, proc_a, pong, MPI_COMM_WORLD, ierror)
        END IF
        
        start = MPI_WTIME()
        
        DO i = 1, number_of_messages
        
           IF (my_rank .EQ. proc_a) THEN
              CALL MPI_SEND(buffer, length, MPI_REAL, proc_b, ping, MPI_COMM_WORLD, ierror)
              CALL MPI_RECV(buffer, length, MPI_REAL, proc_b, pong, MPI_COMM_WORLD, status, ierror)
           ELSE IF (my_rank .EQ. proc_b) THEN
              CALL MPI_RECV(buffer, length, MPI_REAL, proc_a, ping, MPI_COMM_WORLD, status, ierror)
              CALL MPI_SEND(buffer, length, MPI_REAL, proc_a, pong, MPI_COMM_WORLD, ierror)
           END IF
        
        END DO
        
        finish = MPI_WTIME()
        
        IF (my_rank .EQ. proc_a) THEN
        
           time = finish - start
           transfer_time = time / (2 * number_of_messages)
        
           WRITE(*,*) INT(length*size_of_real),'bytes  ', transfer_time*1e6,'usec  ', 1e-6*length*size_of_real/transfer_time,'MB/s'
        
        END IF
   
        length = length * length_factor
   
     END DO
   
     CALL MPI_FINALIZE(ierror)
   
   END PROGRAM
   
Or a similar code from `EPCC - University of Edinburgh 
<http://www.archer.ac.uk/training/course-material/2018/07/mpi-epcc/index.php>`_

.. code-block:: fortran

   !
   ! Program in which 2 processes repeatedly pass a message back and forth
   !
   ! The same data is sent from A to B, then returned from B to A.
   !
   
   program pingpong
   implicit none
   include 'mpif.h'
   
   integer :: ierr, size, rank, comm, i, length, numiter
   integer :: status(MPI_STATUS_SIZE)
   integer :: tag1, tag2, extent
   character*10 temp_char10
   integer :: iargc
   real, allocatable :: sbuffer(:)
   double precision :: tstart, tstop, time, totmess
   
   comm = MPI_COMM_WORLD
   tag1 = 1
   tag2 = 2
   
   call MPI_INIT(ierr)
   call MPI_COMM_RANK(comm,rank,ierr)
   call MPI_COMM_SIZE(comm,size,ierr)
   
   if (iargc() /= 2) then
    if (rank .eq. 0) then
      write(*,*) 'Usage: pingpong <array length> <number of iterations>'
    end if
   
    call mpi_finalize(ierr)
    stop
   end if
   
   
   if (rank.gt.1) then
    print*, 'Rank not participating', rank
   end if
   
   
   if (rank .eq. 0) then 
    call getarg(1,temp_char10)
    read(temp_char10,*) length
    call getarg(2,temp_char10)
    read(temp_char10,*) numiter
   
    print*, 'Array length, number of iterations = '
    print*,  length, numiter
   end if
   
   call MPI_BCAST(length,1,MPI_INTEGER,0,comm,ierr)
   call MPI_BCAST(numiter,1,MPI_INTEGER,0,comm,ierr)
   
   ! Must be run on at least 2 processors
   if(size.lt.2)then
    if(rank.eq.0) write(*,*) ' The code must be run on at least 2 processors.'
    call MPI_FINALIZE(ierr)
    stop
   endif
   
   ! Allocate array
   allocate(sbuffer(length))
   
   ! Send 'buffer' back and forth between rank 0 and rank 1.
   do i=1,length
    sbuffer(i) = rank + 10.d0
   enddo
   
   ! Start timing the parallel part here.
   call MPI_BARRIER(comm,ierr)                                           
   tstart = MPI_Wtime()
   
   do i=1,numiter
    if (rank.eq.0)then
     call MPI_SSEND(sbuffer(1),length,MPI_REAL,1,tag1,comm,ierr)
     call MPI_RECV(sbuffer(1),length,MPI_REAL,1,tag2,comm,status,ierr)
   
    else if (rank.eq.1)then
     call MPI_RECV(sbuffer(1),length,MPI_REAL,0,tag1,comm,status,ierr)
     call MPI_SSEND(sbuffer(1),length,MPI_REAL,0,tag2,comm,ierr)
    endif
   enddo
   
   
   tstop = MPI_Wtime()
   time  = tstop - tstart
   
   call MPI_TYPE_SIZE(MPI_REAL,extent,ierr)
   
   if(rank.eq.0)then
    totmess = 2.d0*extent*length/1024.d0*numiter/1024.d0
    write(*,*) ' Ping-Pong of twice ',extent*length,' bytes, for ',numiter,' times.'
    write(*,*) 'Total computing time is ',time,' [s].'
    write(*,*) 'Total message size is ',totmess,' [MB].'
    write(*,*) 'Latency (time per message) is ', time/numiter*0.5d0,'[s].'
    write(*,*) 'Bandwidth (message per time) is ',totmess/time,' [MB/s].'
   
    if(time.lt.1.d0) then
            ! write(*,*) "WARNING! The time is too short to be meaningful, increase the number 
     ! of iterations and/or the array size so time is at least one second!"
       
       
    endif
   endif
   
   deallocate(sbuffer)
   
   call MPI_FINALIZE(ierr)
   
   end program pingpong

Please choose one of these programs and save it to ``pingpong.f90``. The def file 
that we used for ``mpitest`` can be used in this case too. All we need to do is to 
replace  ``mpitest.c /opt`` with ``pingpong.f90 /opt`` at ``%files`` and to change 
the complition at the end of ``%post`` from ``mpicc -o mpitest mpitest.c`` to 
``mpif90 -o pingpong.x pingpong.f90``. You can also directly compile it on the cluster
and copy the binary file ``pingpong.x`` instead of the code. The container creation 
command remains the same.

Similar to ``mpitest``, we run the command 

.. code-block:: bash

  mpirun -n 2 singularity exec mpi_hybrid.sif /opt/pingpong.x

The output should look like

.. code-block:: bash

   message size   transfertime    bandwidth
          32 bytes     1.6736700000000000      usec     19.119659143802402      MB/s
        2048 bytes     2.9812600000000011      usec     686.95786171930536      MB/s
      131072 bytes     19.135579999999994      usec     6849.6486476540076      MB/s
     8388608 bytes     523.77068999999995      usec     16015.802600219577      MB/s

Let's analyze what just happened: 
When the ``mpirun`` is invoked as shown above, the MPI-based application code, 
which will be linked against the MPI libraries, will make MPI API calls into these 
MPI libraries which in turn talk to the MPI daemon process running on the host system. 
This daemon process handles the communication between MPI processes, including talking 
to the daemons on other nodes to exchange information between processes running on 
different machines, as necessary.

Ultimately, this means that our running MPI code is linking to the MPI libraries
from the MPI install within our container and these are, in turn, communicating
with the MPI daemon on the host system which is part of the host system's MPI 
installation. These two installations of MPI may be different but as long as there 
is compatibility between the version of MPI installed in your container image and 
the version on the host system, your job should run successfully.

As a side note, when running code within a Singularity container, we don't use 
the MPI executables stored within the container (i.e. we **DO NOT** run 
``singularity exec mpirun -np <numprocs> /path/to/my/executable``).
Instead we use the MPI installation on the host system to run Singularity and start
an instance of our executable from within a container for each MPI process.

GPU and MPI
-----------
In the :doc:`/hvd_intro`, we discussed how Horovod uses the MPI in conjuction with NCCL
to scale up apps. In this section, we see a simple example of using a similar concept for
running/training an app or a network on GPUs. The main advantage of such scheme is its
possibility of scaling.

In the below CUDA code, a large is divided by the number of available processers. While the
summation over each chunck is done within a GPU, the total sum is calculated using MPI
AllReduce method. Here, we pin (assume) there is one GPU per CPU.

.. code-block:: c++

    #include <mpi.h>
    #include <cstdio>
    #include <chrono>
    #include <iostream>
    
    
    __global__ void kernel (double* x, int N) {
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < N) {
            x[idx] += 1.0;
        }
    }
    
    // naive atomic reduction kernel
    __global__ void atomic_red(const double  *gdata, double *out, int N){
      size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
      if (idx < N) {
        atomicAdd(out, gdata[idx]);
      }
    }
    
    
    int main(int argc, char** argv) {
    
        int rank, num_ranks;
    
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
        // Binding the cuda device with local MPI rank
        int local_rank, local_size;
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,  MPI_INFO_NULL, &local_comm);
    
        MPI_Comm_size(local_comm, &local_size);
        MPI_Comm_rank(local_comm, &local_rank);
        cudaSetDevice(local_rank%local_size); 
    
        // Total problem size
        size_t N = 1024 * 1024 * 1024;
    
        // Problem size per rank (assumes divisibility of N)
        size_t N_per_rank = N / num_ranks;
    
        // Adapt the last mpi_rank if necessary
        if (rank == (num_ranks - 1)) {
          N_per_rank = N - N_per_rank * (num_ranks - 1);
        }
    
        // Initialize d_local_x to zero on device
        double* d_local_x;
        cudaMalloc((void**) &d_local_x, N_per_rank * sizeof(double));
        cudaMemset(d_local_x, 0.0, N_per_rank*sizeof(double));
    
        double *d_local_sum, *h_local_sum;
        h_local_sum = new double;
        cudaMalloc(&d_local_sum, sizeof(double));
            
        // Number of repetitions
        const int num_reps = 100;
    
        using namespace std::chrono;
    
        auto start = high_resolution_clock::now();
    
        int threads_per_block = 256;
        size_t blocks = (N_per_rank + threads_per_block - 1) / threads_per_block;
    
        for (int i = 0; i < num_reps; ++i) {
            kernel<<<blocks, threads_per_block>>>(d_local_x, N_per_rank);
            cudaDeviceSynchronize();
        }
    
        // summarize the vector of d_x
        atomic_red<<<blocks, threads_per_block>>>(d_local_x, d_local_sum, N_per_rank);
         
        auto end = high_resolution_clock::now();
    
        auto duration = duration_cast<milliseconds>(end - start);
    
        // Copy vector sums from device to host:
        cudaMemcpy(h_local_sum, d_local_sum, sizeof(double), cudaMemcpyDeviceToHost);
    
        // Reduce all sums into the global sum
        double h_global_sum;
        MPI_Allreduce(h_local_sum, &h_global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        std::cout << "Time per kernel = " << duration.count() << " ms " << std::endl;
    
        if (rank == 0) {
          if (abs(h_global_sum - N*100) > 1e-14) {
            std::cerr << "The sum is incorrect!" << std::endl;
            return -1;
          }
          std::cout << "The total sum of x = " << h_global_sum << std::endl;
        }
    
        MPI_Finalize();
    
        return 0;
    }


Please save this as ``reduction.cu`` and compile the code using the command

.. code-block:: bash

   module add OpenMPI/4.0.5-gcccuda-2020b
   nvcc -arch=sm_80 -o reduction.x reduction.cu -I/cvmfs/sling.si/modules/el7/software/OpenMPI/4.0.5-gcccuda-2020b/include -L/cvmfs/sling.si/modules/el7/software/hwloc/2.2.0-GCCcore-10.2.0/lib -lmpi -lcudart

Afterwards, you can use the definition file given below to create the desirable contianer.
Since we will use a similar container for the last section, more details about the definition 
file will be given in below.

.. code-block:: dockerfile

   BootStrap: docker
   From: nvidia/cuda:11.1.1-devel-ubuntu18.04
   
   %files
       reduction.x /
   
   %environment
       # Point to OMPI binaries, libraries, man pages
       export OMPI_DIR=/opt/ompi
       export PATH="$OMPI_DIR/bin:$PATH"
       export LD_LIBRARY_PATH="$OMPI_DIR/lib:$LD_LIBRARY_PATH"
       export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
       export MANPATH="$OMPI_DIR/share/man:$MANPATH"
       export LC_ALL=C
       export HOROVOD_GPU_ALLREDUCE=NCCL
       export HOROVOD_GPU_ALLGATHER=MPI
       export HOROVOD_GPU_BROADCAST=MPI
       export HOROVOD_NCCL_HOME=/usr/local/cuda/nccl
       export HOROVOD_NCCL_INCLUDE=/usr/local/cuda/nccl/include
       export HOROVOD_NCCL_LIB=/usr/local/cuda/nccl/lib 
       export PYTHON_VERSION=3.7
       export TENSORFLOW_VERSION=2.7.0
       export CUDNN_VERSION=8.0.4.30-1+cuda11.1
       export NCCL_VERSION=2.8.3-1+cuda11.0
   
   %post
       mkdir /data1 /data2 /data0
       mkdir -p /var/spool/slurm
       mkdir -p /d/hpc
       mkdir -p /ceph/grid
       mkdir -p /ceph/hpc
       mkdir -p /scratch
       mkdir -p /exa5/scratch
   
       export PYTHON_VERSION=3.7
       export TENSORFLOW_VERSION=2.7
       export CUDNN_VERSION=8.0.4.30-1+cuda11.1
       export NCCL_VERSION=2.8.3-1+cuda11.0
   
       echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
   
       apt-get -y update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
           build-essential \
           cmake \
           git \
           curl \
           vim \
           wget \
           ca-certificates \
           libcudnn8=${CUDNN_VERSION} \
           libnccl2=${NCCL_VERSION} \
           libnccl-dev=${NCCL_VERSION} \
           libjpeg-dev \
           libpng-dev \
           python${PYTHON_VERSION} \
           python${PYTHON_VERSION}-dev \
           python${PYTHON_VERSION}-distutils
   
       ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
   
       curl -O https://bootstrap.pypa.io/get-pip.py && \
       python get-pip.py && \
       rm get-pip.py
   
   # Install Open MPI
       echo "Installing Open MPI"
       export OMPI_DIR=/opt/ompi
       export OMPI_VERSION=4.0.5
       export OMPI_URL="https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-$OMPI_VERSION.tar.bz2"
       mkdir -p /tmp/ompi
       mkdir -p /opt
       # Download
       cd /tmp/ompi && wget -O openmpi-$OMPI_VERSION.tar.bz2 $OMPI_URL && tar -xjf openmpi-$OMPI_VERSION.tar.bz2
       # Compile and install
       cd /tmp/ompi/openmpi-$OMPI_VERSION && ./configure --prefix=$OMPI_DIR && make -j8 install
   
       # Set env variables so we can compile our application
       export PATH=$OMPI_DIR/bin:$PATH
       export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH
   
   # Install TensorFlow, Keras
       pip install tensorflow-gpu==${TENSORFLOW_VERSION} h5py tensorflow-hub
   
   # Install the IB verbs
       apt install -y --no-install-recommends libibverbs*
       apt install -y --no-install-recommends ibverbs-utils librdmacm* infiniband-diags libmlx4* libmlx5* libnuma*
   
   # Install Horovod, temporarily using CUDA stubs
       ldconfig /usr/local/cuda-11.1/targets/x86_64-linux/lib/stubs && \
       HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=0 pip install --no-cache-dir horovod && \
       ldconfig
   
   # Configure OpenMPI to run good defaults:
   #   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
       echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
       echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf 
       #echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf
   
   # Set default NCCL parameters
       echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
       echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

Saving it as ``cuda_example.def``, we can create the ``cuda_example.sif`` as mentioned above. 
Similarly, we can run our example using

.. code-block:: bash

   mpirun -n 4 singularity exec --nv cuda_example.sif /reduction.x

We should see an output similar to

.. code-block:: bash

   Time per kernel = 1676 ms 
   The total sum of x = 1.07374e+11
   Time per kernel = 1691 ms 
   Time per kernel = 1689 ms 
   Time per kernel = 1581 ms 

This example shows the simplest way of *offloading* a job to GPU(s) and using
the MPI AllReduce was used to calculate the final value. The example above can mimic 
the calculation of gradient across difference GPUs.

Training an NLP model using Horovod
-----------------------------------

For the final part, let's train the NLP model we used in previous chapters using containers. 
Since we assume that the cluster does not provide TensorFlow and Horovod
for our training, we don't need to load these two modules for the rest of our work.
We have the option either copying our code and dataset to the container or binding the current
path to ``singularity`` so that it can read file and folders. So far, we avoided 
the latter because it interferes with building the containers with created above.
To keep the same tradition let's copy the code and dataset to the container as we did in other section by
adding the ``Transfer_Learning_NLP_Horovod.py`` code and dataset ``dataset.pkl`` from :doc:`/hvd_intro` 
to a new folder ``horovod`` and adding that to ``%files`` section.
After creating the container we are ready to to traino our model on two processers 
using the command

.. code-block:: bash

   mpirun -n 2 -H localhost:2 singularity exec --nv horovod.sif python horovod/Transfer_Learning_NLP_Horovod.py


.. code-block:: bash

   --------------------------------------------------------------------------
   By default, for Open MPI 4.0 and later, infiniband ports on a device
   are not used by default.  The intent is to use UCX for these devices.
   You can override this policy by setting the btl_openib_allow_ib MCA parameter
   to true.
   
     Local host:              vglogin0008
     Local adapter:           mlx5_0
     Local port:              1
   
   --------------------------------------------------------------------------
   --------------------------------------------------------------------------
   WARNING: There was an error initializing an OpenFabrics device.
   
     Local host:   vglogin0008
     Local device: mlx5_0
   --------------------------------------------------------------------------
   Version:  2.7.0
   Hub version:  0.12.0
   GPU is available
   Number of GPUs : 1
   The shape of training (653061, 3) and validation (653, 3) datasets.
   ##-------------------------##
   
   ##-------------------------##
   Training starts ...
   Epoch 1/40
       1/20408 [..............................] - ETA: 18:55:44 - loss: 0.6903 - accuracy: 0.5938

There is whole host of flags at our disposal which can be/must be used to successfully train 
the network. For example

.. code-block:: bash
  
   mpirun -np 4 -H localhost:4 -x LD_LIBRARY_PATH -x PATH -x HOROVOD_MPI_THREADS_DISABLE=1 -x NCCL_SOCKET_IFNAME=^virbr0,lo -mca btl openib,self -mca pml ob1 singularity exec --nv horovod.sif python /horovod/Transfer_Learning_NLP_Horovod.py

It is always recommended to consult with the system admin regarding the usage of such
flags since it all depends on how the MPI and rest of system is setup.

.. exercise:: What is in the definition file?

   The definition file for the CUDA example and Horovod training is
   almost the same. Can you go through the file explain what each part does?
