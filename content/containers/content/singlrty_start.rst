.. _singlrty_start:

What is Singularity?
--------------------

`Singularity <https://sylabs.io/singularity/>`_ is another container
platform. In some ways it appears similar to Docker from a user
perspective, but in others, particularly in the system's architecture,
it is fundamentally different. These differences mean that Singularity
is particularly well-suited to running on distributed, High
Performance Computing (HPC) infrastructure, as well as a Linux laptop
or desktop!

System administrators will not, generally, install Docker on shared
computing platforms such as lab desktops, research clusters or HPC
platforms because the design of Docker presents potential security
issues for shared platforms with multiple users. Singularity, on the
other hand, can be run by end-users entirely within "user space", that
is, no special administrative privileges need to be assigned to a user
in order for them to run and interact with containers on a platform
where Singularity has been installed.

Getting started with Singularity
--------------------------------

Initially developed within the research community, Singularity is open
source and the `repository <https://github.com/hpcng/singularity>`_ is
currently available in the "`The Next Generation of High Performance
Computing <https://github.com/hpcng>`_" GitHub organisations.  Part I
of this Singularity material is intended to be undertaken on a remote
platform where Singularity has been pre-installed.

If you're attending a taught version of this course, you will be
provided with access details for a remote platform made available to
you for use for Part I of the Singularity material. This platform will
have the Singularity software pre-installed.

.. callout:: Installing Singularity on your own laptop/desktop

   If you have a Linux system on which you have administrator access
   and you can install Singularity locally on this system.

Sign in to the Vega system that you've been provided with access to.
Check that the `singularity` command is available in your terminal:

.. code-block:: bash

   singularity --version

Output

.. code-block:: bash

   SingularityPRO version 3.7-5.el8

Depending on the version of Singularity installed on your system, you
may see a different version.

.. callout:: Singularity on HPC systems: Loading a module

   HPC systems often use *modules* to provide access to software on the
   system.  If you get a command not found error (e.g. `bash:
   singularity: command not found` or similar) you may need to load the
   *singularity module* before you can use the `singularity` command:

   .. code-block:: bash

      module load singularity

Nonetheless, for the Vega system `singularity` is readily available.


Images and containers
---------------------

We'll start with a brief note on the terminology used in this section
of the course.  We refer to both **images** and **containers**. What
is the distinction between these two terms?

**Images** are bundles of files including an operating system,
software and potentially data and other application-related
files. They may sometimes be referred to as a disk image or container
image and they may be stored in different ways, perhaps as a single
file, or as a group of files.  Either way, we refer to this file, or
collection of files, as an image.

A **container** is a virtual environment that is based on an
image. That is, the files, applications, tools, etc that are available
within a running container are determined by the image that the
container is started from. It may be possible to start multiple
container instances from an image. You could, perhaps, consider an
image to be a form of template from which running container instances
can be started.

Getting an image and running a Singularity container
----------------------------------------------------

If you recall from learning about Docker, Docker images are formed of
a set of layers that make up the complete image. When you pull a
Docker image from Docker Hub, you see the different layers being
downloaded to your system. They are stored in your local Docker
repository on your system and you can see details of the available
images using the `docker` command.

Singularity images are a little different. Singularity uses the
`Signularity Image Format (SIF) <https://github.com/sylabs/sif>`_ and
images are provided as single `SIF` files. Singularity images can be
pulled from `Singularity Hub <https://singularity-hub.org/>`_, a
registry for container images. Singularity is also capable of running
containers based on images pulled from `Docker Hub
<https://hub.docker.com/>`_ and some other sources. We'll look at
accessing containers from Docker Hub later in the Singularity
material.

.. callout:: Singularity Hub

   Note that in addition to providing a repository that you can pull
   images from, `Singularity Hub <https://singularity-hub.org/>`_ can
   also build Singularity images for you from a `recipe` - a
   configuration file defining the steps to build an image.  We'll look
   at recipes and building images later.

Let's begin by creating a `test` directory, changing into it and
pulling a test Hello World image from Singularity Hub:

.. code-block:: bash

   mkdir test
   cd test
   singularity pull hello-world.sif shub://vsoch/hello-world

.. note::

  What is the main difference between above and the Docker pull request?


Output

.. code-block:: bash

   INFO:    Downloading shub image
   59.75 MiB / 59.75 MiB [=====================================================================] 100.00% 52.03 MiB/s 1s


What just happened?! We pulled a SIF image from Singularity Hub using
the `singularity pull` command and directed it to store the image file
using the name `hello-world.sif`. If you run the `ls` command, you
should see that the `hello-world.sif` file is now in your current
directory. This is our image and we can now run a container based on
this image:

.. code-block:: bash

   singularity run hello-world.sif

Output

.. code-block:: bash

   RaawwWWWWWRRRR!! Avocado!


The above command ran the hello-world container from the image we
downloaded from Singularity Hub and the resulting output was shown.


How did the container determine what to do when we ran it?! What did
running the container actually do to result in the displayed output?

When you run a container from an image without using any additional
command line arguments, the container runs the default run script that
is embedded within the image. This is a shell script that can be used
to run commands, tools or applications stored within the image on
container startup. We can inspect the image's run script using the
`singularity inspect` command:

.. code-block:: bash

   singularity inspect -r hello-world.sif


Output

.. code-block:: bash

   #!/bin/sh

   exec /bin/bash /rawr.sh

This shows us the script within the `hello-world.sif` image configured
to run by default when we use the ``singularity run`` command.

That concludes this introductory Singularity episode. The next episode
looks in more detail at running containers.
