.. _build_contain:

Building Singularity images
===========================

Preparing to use Singularity for building images
------------------------------------------------

So far you've been able to work with Singularity from your own user
account as a non-privileged user.  This part of the Singularity
material requires that you use Singularity in an environment where you
have administrative (root) access. While it is possible to build
Singularity containers without root access, it is highly recommended
that you do this as the **root** user, as highlighted in `this section
<https://sylabs.io/guides/3.7/user-guide/build_a_container.html#creating-writable-sandbox-directories>`_
of the Singularity documentation. Bear in mind that the system that
you use to build containers doesn't have to be the system where you
intend to run the containers. If, for example, you are intending to
build a container that you can subsequently run on a Linux-based
cluster, you could build the container on your own Linux-based desktop
or laptop computer. You could then transfer the built image directly
to the target platform or upload it to an image repository and pull it
onto the target platform from this repository.

There are three different options for accessing a suitable environment
to undertake the material in this part of the course:

 1. Run Singularity from within a Docker container - this will enable you to have the required privileges to build images
 2. Install Singularity locally on a system where you have administrative access
 3. Use Singularity on a system where it is already pre-installed and you have administrative (root) access

We'll focus on the first option in this part of the course. If you
would like to install Singularity directly on your system, see the box
below for some further pointers. Note that the installation process is
an advanced task that is beyond the scope of this course so we won't
be covering this.

.. note:: Installing Singularity on your local system (optional)

   If you are running Linux and would like to install Singularity
   locally on your system, Singularity provide the free, open source
   `Singularity Community Edition
   <https://github.com/hpcng/singularity/releases>`_. You will need to
   install various dependencies on your system and then build
   Singularity from source code.

   If you are not familiar with building applications from source code,
   it is strongly recommended that you use the Docker Singularity
   image, as described below in the "Getting started with the Docker
   Singularity image" section rather than attempting to build and
   install Singularity yourself. The installation process is an
   advanced task that is beyond the scope of this session.

   However, if you have Linux systems knowledge and would like to
   attempt a local install of Singularity, you can find details in the
   `INSTALL.md
   <https://github.com/sylabs/singularity/blob/master/INSTALL.md>`_
   file within the Singularity repository that explains how to install
   the prerequisites and build and install the software.  Singularity
   is written in the `Go <https://golang.org/>`_ programming language and
   Go is the main dependency that you'll need to install on your
   system. The process of installing Go and any other requirements is
   detailed in the INSTALL.md file.


.. tip::

   If you do not have access to a system with Docker installed, or a
   Linux system where you can build and install Singularity but you
   have administrative privileges on another system, you could look at
   installing a virtualisation tool such as `VirtualBox
   <https://www.virtualbox.org/>`_ on which you could run a Linux
   Virtual Machine (VM) image. Within the Linux VM image, you will be
   able to install Singularity.  Again this is beyond the scope of the
   course.

   If you are not able to access/run Singularity yourself on a system
   where you have administrative privileges, you can still follow
   through this material as it is being taught (or read through it in
   your own time if you're not participating in a taught version of the
   course) since it will be helpful to have an understanding of how
   Singularity images can be built.

   You could also attempt to follow this section of the lesson without
   using root and instead using the ``singularity`` command's `--fakeroot
   <https://sylabs.io/guides/3.7/user-guide/fakeroot.html>`_ option.
   However, you may encounter issues with permissions when trying to
   build images and run your containers and this is why running the
   commands as root is strongly recommended and is the approach
   described in this lesson.

Getting started with the Docker Singularity image
-------------------------------------------------

The `Singularity Docker image <https://quay.io/repository/singularity/singularity>`_ is available from
`Quay.io <https://quay.io/>`_.

.. exercise:: Familiarise yourself with the Docker Singularity image

  - Using your previously acquired Docker knowledge, get the
    Singularity image for ``v3.8.2`` and ensure that you can run a Docker
    container using this image. You might want to use the `v3.8.2-slim` for Intel/AMD architecture
    or `v3.8.2-slim-arm64` for Arm architecture version of this image since it is significantly 
    smaller than the standard image - the **slim** version of the image will be used in the
    examples below.

  - Create a directory (e.g. ``$HOME/singularity_data``) on your host
    machine that you can use for storage of **definition files** (we'll
    introduce these shortly) and generated image files.

    This directory should be bind mounted into the Docker container at
    the location `/home/singularity` every time you run it - this will
    give you a location in which to store built images so that they are
    available on the host system once the container exits.  (take a look
    at the ``-v`` switch)

    **Note**: To be able to build an image using the Docker Singularity
    container, you'll probably need to add the ``--privileged`` switch to
    your docker command line.

	  - What is happening when you run the container?
	  - Can you run an interactive shell in the container?

       .. solution:: Running the image
         
         Having a bound directory from the host system accessible within
	      your running Singularity container will give you somewhere to
	      place created images so that they are accessible on the host
	      system after the container exits.  Begin by changing into the
	      directory that you created above for storing your definiton
	      files and built images (e.g. ``$HOME/singularity_data``).
    
	      You may choose to:
    
	      - open a shell within the Docker image so you can work at a
	        command prompt and run the ``singularity`` command directly
	      - use the ``docker run`` command to run a new container instance
	        every time you want to run the `singularity` command.
    
	      Either option is fine for this section of the material.
    
	      **Some examples:**
    
	      To run the ``singularity`` command within the docker container
	      directly from the host system's terminal:
    
	      .. code-block:: bash
    
	        docker run -it --privileged --rm -v ${PWD}:/home/singularity
	        quay.io/singularity/singularity:v3.8.2-slim cache list
    
	      To start a shell within the Singularity Docker container where
	      the `singularity` command can be run directly:
    
	      .. code-block:: bash
    
	         docker run -it --entrypoint=/bin/sh --privileged --rm -v ${PWD}:/home/singularity quay.io/singularity/singularity:v3.8.2-slim
    
	      To make things easier to read in the remainder of the material,
	      command examples will use the ``singularity`` command directly,
	      e.g. ``singularity cache list``. If you're running a shell in the
	      Docker container, you can enter the commands as they appear.  If
	      you're using the container's default run behavior and running a
	      container instance for each run of the command, you'll need to
	      replace ``singularity`` with ``docker run --privileged -v
	      ${PWD}:/home/singularity quay.io/singularity/singularity:v3.8.2-slim`` or similar.

Building Singularity images
---------------------------

Introduction
++++++++++++

As a platform that is widely used in the scientific/research software and HPC communities, 
Singularity provides great support for reproducibility.
If you build a Singularity container for some scientific software, it's likely that you and/or 
others will want to be able to reproduce exactly
the same environment again. Maybe you want to verify the results of the code or provide a means 
that others can use to verify the results to support a paper or report.
Maybe you're making a tool available to others and want to ensure that they have exactly the right 
version/configuration of the code.

Similarly to Docker and many other modern software tools, Singularity
follows the "Configuration as code" approach and a container
configuration can be stored in a file which can then be committed to
your version control system alongside other code. Assuming it is
suitably configured, this file can then be used by you or other
individuals (or by automated build tools) to reproduce a container
with the same configuration at some point in the future.

Different approaches to building images
+++++++++++++++++++++++++++++++++++++++

There are various approaches to building Singularity images. We
highlight two different approaches here and focus on one of them:

- Building within a sandbox: You can build a container
  interactively within a `sandbox environment
  <https://sylabs.io/guides/3.7/user-guide/build_a_container.html#creating-writable-sandbox-directories>`_.
  This means you get a shell within the container environment and
  install and configure packages and code as you wish before exiting the
  sandbox and converting it into a container image.


- Building from a `Singularity Definition File
  <https://sylabs.io/guides/3.7/user-guide/build_a_container.html#building-containers-from-singularity-definition-files>`_:
  This is Singularity's equivalent to building a Docker container from a
  `Dockerfile` and we'll discuss this approach in this section.

You can take a look at Singularity's "`Build a Container
<https://sylabs.io/guides/3.7/user-guide/build_a_container.html>`_"
documentation for more details on different approaches to building
containers.

.. exercise:: Why look at Singularity Definition Files?

   Why do you think we might be looking at the definition file
   approach here rather than the *sandbox approach*?

   .. solution::

      The sandbox approach is great for prototyping and testing out an
      image configuration but it doesn't provide the best support for
      our ultimate goal of **reproducibility**. If you spend time
      sitting at your terminal in front of a shell typing different
      commands to add configuration, maybe you realize you made a
      mistake so you undo one piece of configuration and change
      it. This goes on until you have your completed configuration but
      there's no explicit record of exactly what you did to create
      that configuration.

      Say your container image file gets deleted by accident, or
      someone else wants to create an equivalent image to test
      something.  How will they do this and know for sure that they
      have the same configuration that you had?  With a definition
      file, the configuration steps are explicitly defined and can be
      easily stored, for example within a version control system, and
      re-run.

      Definition files are small text files while container files may
      be very large, multi-gigabyte files that are difficult and time
      consuming to move around. This makes definition files ideal for
      storing in a version control system along with their revisions.

Creating a Singularity Definition File
++++++++++++++++++++++++++++++++++++++

A Singularity Definition File is a text file that contains a series of statements that are used to create a container image.
In line with the *configuration as code* approach mentioned above, the definition file can be stored in your code repository
alongside your application code and used to create a reproducible image. This means that for a given commit in your repository,
the version of the definition file present at that commit can be used to reproduce a container with a known state.
It was pointed out earlier in the course, when covering Docker, that this property also applies for Dockerfiles.

We'll now look at a very simple example of a definition file:

.. code-block:: bash

  Bootstrap: docker
  From: ubuntu:20.04

  %post
    apt-get -y update && apt-get install -y python3

  %runscript
    python3 -c 'print("Hello World! Hello from our custom Singularity image!")'

A definition file has a number of optional sections, specified using the `%` prefix,
that are used to define or undertake different configuration during different stages of the image build process.
You can find full details in Singularity's `Definition Files documentation <https://sylabs.io/guides/3.7/user-guide/definition_files.html>`_.
In our very simple example here, we only use the `%post` and `%runscript` sections.

Let's step through this definition file and look at the lines in more detail:

.. code-block:: bash

  Bootstrap: docker
  From: ubuntu:20.04


These first two lines define where to **bootstrap** our image from. Why can't we just put some application binaries into
a blank image? Any applications or tools that we want to run will need to interact with standard system libraries and
potentially a wide range of other libraries and tools. These need to be available within the image and we therefore
need some sort of operating system as the basis for our image. The most straightforward way to achieve this is to start
from an existing base image containing an operating system. In this case, we're going to start from a minimal Ubuntu 20.04
Linux Docker image. Note that we're using a Docker image as the basis for creating a Singularity image.
This demonstrates the flexibility in being able to start from different types of images when creating a new Singularity image.

The ``Bootstrap: docker`` line is similar to prefixing an image path with ``docker://`` when using, for example,
the ``singularity pull`` command. A range of `different bootstrap options <https://sylabs.io/guides/3.7/user-guide/definition_files.html#preferred-bootstrap-agents>`_
are supported. ``From: ubuntu:20.04`` says that we want to use the ``ubuntu`` image with the tag ``20.04``.

Next we have the `%post` section of the definition file:

.. code-block:: bash

  %post
    apt-get -y update && apt-get install -y python3

In this section of the file we can do tasks such as package installation, pulling data files from remote locations
and undertaking local configuration within the image. The commands that appear in this section are standard shell
commands and they are run **within** the context of our new container image. So, in the case of this example,
these commands are being run within the context of a minimal Ubuntu 20.04 image that initially has only a very small
set of core packages installed.

Here we use Ubuntu's package manager to update our package indexes and then install the ``python3`` package along
with any required dependencies (in Ubuntu 20.04, the **python3** package installs ``python 3.8.5``). The ``-y`` switches
are used to accept, by default, interactive prompts that might appear asking you to confirm package updates or installation.
This is required because our definition file should be able to run in an unattended, non-interactive environment.

Finally we have the ``%runscript`` section:

.. code-block:: bash

  %runscript
    python3 -c 'print("Hello World! Hello from our custom Singularity image!")'

This section is used to define a script that should be run when a container is started based on this image using
the `singularity run` command. In this simple example we use `python3` to print out some text to the console.

We can now save the contents of the simple defintion file shown above to a file and build an image based on it.
In the case of this example, the definition file has been named `my_test_image.def`. (Note that the instructions
here assume you've bound the image output directory you created to the `/home/singularity` directory in your Docker Singularity container):

.. code-block:: bash

  singularity build /home/singularity/my_test_image.sif /home/singularity/my_test_image.def

Recall from the details at the start of this section that if you are running your command from the host system command line,
running an instance of a Docker container for each run of the command, your command will look something like this:

.. code-block:: bash

  docker run -it --privileged --rm -v ${PWD}:/home/singularity quay.io/singularity/singularity:v3.8.2-slim build /home/singularity/my_test_image.sif /home/singularity/my_test_image.def

The above command requests the building of an image based on the `my_test_image.def` file with the resulting image
saved to the `my_test_image.sif` file. Note that you will need to prefix the command with `sudo` if you're running
a locally installed version of Singularity and not running via Docker because it is necessary to have administrative
privileges to build the image. You should see output similar to the following:

.. code-block:: text

  INFO:    Starting build...
  Getting image source signatures
  Copying blob da7391352a9b done
  Copying blob 14428a6d4bcd done
  Copying blob 2c2d948710f2 done
  Copying config aa23411143 done
  Writing manifest to image destination
  Storing signatures
  2020/12/08 09:15:18  info unpack layer: sha256:da7391352a9bb76b292a568c066aa4c3cbae8d494e6a3c68e3c596d34f7c75f8
  2020/12/08 09:15:19  info unpack layer: sha256:14428a6d4bcdba49a64127900a0691fb00a3f329aced25eb77e3b65646638f8d
  2020/12/08 09:15:19  info unpack layer: sha256:2c2d948710f21ad82dce71743b1654b45acb5c059cf5c19da491582cef6f2601
  INFO:    Running post scriptlet
  + apt-get -y update
  Get:1 http://archive.ubuntu.com/ubuntu focal InRelease [265 kB]
  ...
  [Package update output truncated]
  ...
  Fetched 16.6 MB in 3s (6050 kB/s)
  Reading package lists...
  + apt-get install -y python3
  Reading package lists...
  ...
  [Package install output truncated]
  ...
  Processing triggers for libc-bin (2.31-0ubuntu9.1) ...
  INFO:    Adding runscript
  INFO:    Creating SIF file...
  INFO:    Build complete: my_test_image.sif
  $


You should now have a ``my_test_image.sif`` file in the current directory. Note that in
your version of the above output, after it says ``INFO:  Starting build...`` you may see
a series of ``skipped: already exists`` messages for the ``Copying blob`` lines. This happens
when the Docker image slices for the Ubuntu 20.04 image have previously been downloaded and
are cached on the system where this example is being run. On your system, if the image is not
already cached, you will see the slices being downloaded from Docker Hub when these lines of output appear.

Permissions of the created image file
+++++++++++++++++++++++++++++++++++++

You may find that the created Singularity image file on your host filesystem is owned by the `root` user and not your user.
In this case, you won't be able to change the ownership/permissions of the file directly if you don't have root access.
However, the image file will be readable by you and you should be able to take a copy of the file under a new name which
you will then own. You will then be able to modify the permissions of this copy of the image and delete the original
root-owned file since the default permissions should allow this.

**Testing your Singularity image**

In a moment we'll test the created image on our HPC platform but, first, you should be able to run a shell in an instance of
the Docker Singularity container and run your singularity image there.

.. exercise:: Run the Singularity image you've created

   Can you run the Singularity image you've just built from a shell
   within the Docker Singularity container?

   .. solution::

      .. code-block:: bash

         docker run -it --entrypoint=/bin/sh --privileged --rm -v ${PWD}:/home/singularity quay.io/singularity/singularity:v3.8.2-slim
         /# cd /home/singularity
         /home/singularity# singularity run my_test_image.sif

      Output

      .. code-block:: text

         Hello World! Hello from our custom Singularity image!
         /home/singularity#

.. callout:: Using ``singularity run`` from within the Docker container

  It is strongly recommended that you don't use the Docker container for running Singularity images
  in any production setting, only for creating them, since the Singularity command runs within the 
  container as the root user.
  However, for the purposes of this simple example, the Docker Singularity container provides an ideal 
  environment to test that
  you have successfully built your container.

Now we'll test our image on an HPC platform. Move your created ``.sif`` image file to a platform with 
an installation of Singularity.
You could, for example, do this using the command line secure copy command ``scp``. For example, 
the following command would copy
`my_test_image.sif` to the remote server identified by ``<target hostname>`` (don't forget 
the colon at the end of the hostname!):

.. code-block:: bash

  scp -i <full path to SSH key file> my_test_image.sif <target hostname>:


You could provide a destination path for the file straight after the colon at the end of the above 
command (without a space),
but by default, the file will be uploaded to you home directory.

Try to run the container on the login node of the HPC platform and check that you get the expected output.

It is recommended that you move the create `.sif` file to a platform with an installation of Singularity, 
rather than attempting to run
the image using the Docker container. However, if you do try to use the Docker container, 
see the notes below on "*Using singularity run from within the Docker container*" for further information.

Now that we've built an image, we can attempt to run it:

.. code-block:: bash

   singularity run my_test_image.sif

If everything worked successfully, you should see the message printed
by Python:

.. code-block:: bash

   Hello World! Hello from our custom Singularity image!

.. callout:: Using ``singularity run`` from within the Docker container

   It is strongly recommended that you don't use the Docker container
   for running Singularity images, only for creating then, since the
   Singularity command runs within the container as the root
   user. However, for the purposes of this simple example, if you are
   trying to run the container using the ``singularity`` command from
   within the Docker container, it is likely that you will get an error
   relating to ``/etc/localtime`` similar to the following:

   .. code-block:: text

      WARNING: skipping mount of /etc/localtime: no such file or directory
      FATAL: container creation failed: mount
      /etc/localtime->/etc/localtime error: while mounting
      /etc/localtime: mount source /etc/localtime doesn't exist

   This occurs because the ``/etc/localtime`` file that provides
   timezone configuration is not present within the Docker container.
   If you want to use the Docker container to test that your newly
   created image runs, you'll need to open a shell in the Docker
   container and add a timezone configuration as described in the
   `Alpine Linux documentation
   <https://wiki.alpinelinux.org/wiki/Setting_the_timezone>`_:

   .. code-block:: bash

      apk add tzdata
      cp /usr/share/zoneinfo/Europe/London /etc/localtime

The ``singularity run`` command should now work successfully.

More about definiton files
--------------------------

A {Singularity} Definition file is divided into two parts:

#. **Header**: The Header describes the core operating system to build within
   the container. Here you will configure the base operating system features
   needed within the container. You can specify, the Linux distribution, the
   specific version, and the packages that must be part of the core install
   (borrowed from the host system).

#. **Sections**: The rest of the definition is comprised of sections, (sometimes
   called scriptlets or blobs of data). Each section is defined by a ``%``
   character followed by the name of the particular section. All sections are
   optional, and a def file may contain more than one instance of a given
   section. Sections that are executed at build time are executed with the
   ``/bin/sh`` interpreter and can accept ``/bin/sh`` options. Similarly,
   sections that produce scripts to be executed at runtime can accept options
   intended for ``/bin/sh``

For more in-depth and practical examples of def files, see the `Singularity examples
repository <https://github.com/hpcng/singularity/tree/master/examples>`_

For a comparison between Dockerfile and {Singularity} definition file,
please see: :ref:`this section <sec:deffile-vs-dockerfile>`.


Header
++++++

The header should be written at the top of the def file. It tells {Singularity}
about the base operating system that it should use to build the container. It is
composed of several keywords.

The only keyword that is required for every type of build is ``Bootstrap``.
It determines the *bootstrap agent*  that will be used to create the base
operating system you want to use. For example, the ``library`` bootstrap agent
will pull a container from the `Container Library
<https://cloud.sylabs.io/library>`_ as a base. Similarly, the ``docker``
bootstrap agent will pull docker layers from `Docker Hub
<https://hub.docker.com/>`_ as a base OS to start your image.

The ``Bootstrap`` keyword needs to be the first
entry in the header section.  This breaks compatibility with older versions
that allow the parameters of the header to appear in any order.

Depending on the value assigned to ``Bootstrap``, other keywords may also be
valid in the header. For example, when using the ``library`` bootstrap agent,
the ``From`` keyword becomes valid. Observe the following example for building a
Debian container from the Container Library:

.. code-block:: singularity

    Bootstrap: library
    From: debian:7

A def file that uses an official mirror to install Centos-7 might look like
this:

.. code-block:: singularity

    Bootstrap: yum
    OSVersion: 7
    MirrorURL: http://mirror.centos.org/centos-%{OSVERSION}/%{OSVERSION}/os/$basearch/
    Include: yum

Each bootstrap agent enables its own options and keywords. You can read about
them and see examples in the :ref:`appendix section<buildmodules>`:


**Preferred bootstrap agents**

-  library (images hosted on the `Container Library <https://cloud.sylabs.io/library>`_)

-  docker (images hosted on Docker Hub)

-  build-shub (images hosted on Singularity Hub)

-  oras (images from supporting OCI registries)

-  scratch (a flexible option for building a container from scratch)

**Other bootstrap agents**

-  localimage (images saved on your machine)

-  yum (yum based systems such as CentOS and Scientific Linux)

-  debootstrap (apt based systems such as Debian and Ubuntu)

-  oci (bundle compliant with OCI Image Specification)

-  oci-archive (tar files obeying the OCI Image Layout Specification)

-  docker-daemon (images managed by the locally running docker daemon)

-  docker-archive (archived docker images)

-  arch (Arch Linux)

-  busybox (BusyBox)

-  zypper (zypper based systems such as Suse and OpenSuse)

A general definition
++++++++++++++++++++

The main content of the bootstrap file is broken into sections. Different
sections add different content or execute commands at different times during the
build process. Note that if any command fails, the build process will halt.

Here is an example definition file that uses every available section. We will
discuss each section in turn. It is not necessary to include every section (or
any sections at all) within a def file. Furthermore, multiple sections of the
same name can be included and will be appended to one another during the build
process.

.. code-block:: singularity

    Bootstrap: library
    From: ubuntu:18.04
    Stage: build

    %setup
        touch /file1
        touch ${SINGULARITY_ROOTFS}/file2

    %files
        /file1
        /file1 /opt

    %environment
        export LISTEN_PORT=12345
        export LC_ALL=C

    %post
        apt-get update && apt-get install -y netcat
        NOW=`date`
        echo "export NOW=\"${NOW}\"" >> $SINGULARITY_ENVIRONMENT

    %runscript
        echo "Container was created $NOW"
        echo "Arguments received: $*"
        exec echo "$@"

    %startscript
        nc -lp $LISTEN_PORT

    %test
        grep -q NAME=\"Ubuntu\" /etc/os-release
        if [ $? -eq 0 ]; then
            echo "Container base is Ubuntu as expected."
        else
            echo "Container base is not Ubuntu."
            exit 1
        fi

    %labels
        Author d@sylabs.io
        Version v0.0.1

    %help
        This is a demo container used to illustrate a def file that uses all
        supported sections.

Although the order of the sections in the def file is unimportant, they have
been documented below in the order of their execution during the build process
for logical understanding.

``%setup``
++++++++++

During the build process, commands in the ``%setup`` section are first executed
on the host system outside of the container after the base OS has been installed.
You can reference the container file system with the ``$SINGULARITY_ROOTFS``
environment variable in the ``%setup`` section.

.. note::

    Be careful with the ``%setup`` section! This scriptlet is executed outside
    of the container on the host system itself, and is executed with elevated
    privileges.

Consider the example from the definition file above:

.. code-block:: singularity

    %setup
        touch /file1
        touch ${SINGULARITY_ROOTFS}/file2

Here, ``file1`` is created at the root of the file system **on the host**.
We'll use ``file1`` to demonstrate the usage of the ``%files`` section below.
The ``file2`` is created at the root of the file system **within the
container**.

In later versions of {Singularity} the ``%files`` section is provided as a safer
alternative to copying files from the host system into the container during the
build. Because of the potential danger involved in running the ``%setup``
scriptlet with elevated privileges on the host system during the build, it's
use is generally discouraged.

``%setup`` can be used for exporting environmental variables.

``%files``
++++++++++

The ``%files`` section allows you to copy files into the container with greater
safety than using the ``%setup`` section. Its general form is:

.. code-block:: singularity

    %files [from <stage>]
        <source> [<destination>]
        ...

Each line is a ``<source>`` and ``<destination>`` pair. The ``<source>`` is either:

  1. A valid path on your host system
  2. A valid path in a previous stage of the build

while the ``<destination>`` is always a path into the current container. If the
``<destination>`` path is omitted it will be assumed to be the same as ``<source>``.
To show how copying from your host system works, let's consider the example from
the definition file above:

.. code-block:: singularity

    %files
        /file1
        /file1 /opt

``file1`` was created in the root of the host file system during the ``%setup``
section (see above).  The ``%files`` scriptlet will copy ``file1`` to the root
of the container file system and then make a second copy of ``file1`` within the
container in ``/opt``.

Files can also be copied from other stages by providing the source location in the
previous stage and the destination in the current container.

.. code-block:: singularity

  %files from stage_name
    /root/hello /bin/hello

The only difference in behavior between copying files from your host system and copying them
from previous stages is that in the former case symbolic links are always followed
during the copy to the container, while in the latter symbolic links are preserved.

Files in the ``%files`` section are always copied before the ``%post`` section is
executed so that they are available during the build and configuration process.

``%post``
+++++++++

This section is where you can download files from the internet with tools like ``git``
and ``wget``, install new software and libraries, write configuration files,
create new directories, etc.

Consider the example from the definition file above:

.. code-block:: singularity

    %post
        apt-get update && apt-get install -y netcat
        NOW=`date`
        echo "export NOW=\"${NOW}\"" >> $SINGULARITY_ENVIRONMENT


This ``%post`` scriptlet uses the Ubuntu package manager ``apt`` to update the
container and install the program ``netcat`` (that will be used in the
``%startscript`` section below).

The script is also setting an environment variable at build time.  Note that the
value of this variable cannot be anticipated, and therefore cannot be set during
the ``%environment`` section. For situations like this, the ``$SINGULARITY_ENVIRONMENT``
variable is provided. Redirecting text to this variable will cause it to be
written to a file called ``/.singularity.d/env/91-environment.sh`` that will be
sourced at runtime.

``%test``
+++++++++

The ``%test`` section runs at the very end of the build process to
validate the container using a method of your choice. You can also
execute this scriptlet through the container itself, using the
``test`` command.

Consider the example from the def file above:

.. code-block:: singularity

    %test
        grep -q NAME=\"Ubuntu\" /etc/os-release
        if [ $? -eq 0 ]; then
            echo "Container base is Ubuntu as expected."
        else
            echo "Container base is not Ubuntu."
            exit 1
        fi


This (somewhat silly) script tests if the base OS is Ubuntu. You could
also write a script to test that binaries were appropriately
downloaded and built, or that software works as expected on custom
hardware. If you want to build a container without running the
``%test`` section (for example, if the build system does not have the
same hardware that will be used on the production system), you can do
so with the ``--notest`` build option:

.. code-block:: none

    $ sudo singularity build --notest my_container.sif my_container.def

Running the test command on a container built with this def file yields the
following:

.. code-block:: none

    $ singularity test my_container.sif
    Container base is Ubuntu as expected.

One common use of the ``%test`` section is to run a quick check that
the programs you intend to install in the container are present. If
you installed the program ``samtools``, which shows a usage screen when
run without any options, you might test it can be run with:

.. code-block:: singularity

    %test
        # Run samtools - exits okay with usage screen if installed
        samtools

If ``samtools`` is not successfully installed in the container then the
``singularity test`` will exit with an error such as ``samtools:
command not found``.

Some programs return an error code when run without mandatory
options. If you want to ignore this, and just check the program is
present and can be called, you can run it as ``myprog || true`` in
your test:

.. code-block:: singularity

    %test
        # Run bwa - exits with error code if installed and run without
        # options
        bwa || true

The ``|| true`` means that if the command before it is found but
returns an error code it will be ignored, and replaced with the error
code from ``true`` - which is always ``0`` indicating success.

Because the ``%test`` section is a shell scriptlet, complex tests are
possible. Your scriptlet should usually be written so it will exit
with a non-zero error code if there is a problem during the tests.

Now, the following sections are all inserted into the container filesystem in
single step:

``%environment``
++++++++++++++++

The ``%environment`` section allows you to define environment variables that
will be set at runtime. Note that these variables are not made available at
build time by their inclusion in the ``%environment`` section. This means that
if you need the same variables during the build process, you should also define
them in your ``%post`` section. Specifically:

-  **during build**: The ``%environment`` section is written to a file in the
   container metadata directory. This file is not sourced.

-  **during runtime**: The file in the container metadata directory is sourced.

You should use the same conventions that you would use in a ``.bashrc`` or
``.profile`` file. Consider this example from the def file above:

.. code-block:: singularity

    %environment
        export LISTEN_PORT=12345
        export LC_ALL=C

The ``$LISTEN_PORT`` variable will be used in the ``%startscript`` section
below. The ``$LC_ALL`` variable is useful for many programs (often written in
Perl) that complain when no locale is set.

After building this container, you can verify that the environment variables are
set appropriately at runtime with the following command:

.. code-block:: none

    $ singularity exec my_container.sif env | grep -E 'LISTEN_PORT|LC_ALL'
    LISTEN_PORT=12345
    LC_ALL=C

In the special case of variables generated at build time, you can also add
environment variables to your container in the ``%post`` section.

At build time, the content of the ``%environment`` section is written to a file
called ``/.singularity.d/env/90-environment.sh`` inside of the container.  Text
redirected to the ``$SINGULARITY_ENVIRONMENT`` variable during ``%post`` is
added to a file called ``/.singularity.d/env/91-environment.sh``.

At runtime, scripts in ``/.singularity/env`` are sourced in order. This means
that variables in the ``%post`` section take precedence over those added  via
``%environment``.

See :ref:`Environment and Metadata <environment-and-metadata>` for more
information about the {Singularity} container environment.

``%runscript``
++++++++++++++

The contents of the ``%runscript`` section are written to a file within the
container that is executed when the container image is run (either via the
``singularity run`` command or by executing the container directly as a
command). When the container is invoked, arguments following the container name
are passed to the runscript. This means that you can (and should) process
arguments within your runscript.

Consider the example from the def file above:

.. code-block:: singularity

    %runscript
        echo "Container was created $NOW"
        echo "Arguments received: $*"
        exec echo "$@"

In this runscript, the time that the container was created is echoed via the
``$NOW`` variable (set in the ``%post`` section above). The options passed to
the container at runtime are printed as a single string (``$*``) and then they
are passed to echo via a quoted array (``$@``) which ensures that all of the
arguments are properly parsed by the executed command. The ``exec`` preceding
the final ``echo`` command replaces the current entry in the process table
(which originally was the call to {Singularity}). Thus the runscript shell process
ceases to exist, and only the process running within the container remains.

Running the container built using this def file will yield the following:

.. code-block:: none

    $ ./my_container.sif
    Container was created Thu Dec  6 20:01:56 UTC 2018
    Arguments received:

    $ ./my_container.sif this that and the other
    Container was created Thu Dec  6 20:01:56 UTC 2018
    Arguments received: this that and the other
    this that and the other


``%labels``
+++++++++++

The ``%labels`` section is used to add metadata to the file
``/.singularity.d/labels.json`` within your container. The general format is a
name-value pair.

Consider the example from the def file above:

.. code-block:: singularity

    %labels
        Author d@sylabs.io
        Version v0.0.1
        MyLabel Hello World


Note that labels are defined by key-value pairs. To define a label just add it
on the labels section and after the first space character add the correspondent value to the label.

In the previous example, the first label name is ``Author``` with a
value of ``d@sylabs.io``. The second label name is ``Version`` with a value of ``v0.0.1``.
Finally, the last label named ``MyLabel`` has the value of ``Hello World``.

To inspect the available labels on your image you can do so by running the following command:

.. code-block:: none

    $ singularity inspect my_container.sif

    {
      "Author": "d@sylabs.io",
      "Version": "v0.0.1",
      "MyLabel": "Hello World",
      "org.label-schema.build-date": "Thursday_6_December_2018_20:1:56_UTC",
      "org.label-schema.schema-version": "1.0",
      "org.label-schema.usage": "/.singularity.d/runscript.help",
      "org.label-schema.usage.singularity.deffile.bootstrap": "library",
      "org.label-schema.usage.singularity.deffile.from": "ubuntu:18.04",
      "org.label-schema.usage.singularity.runscript.help": "/.singularity.d/runscript.help",
      "org.label-schema.usage.singularity.version": "3.0.1"
    }

Some labels that are captured automatically from the build process. You can read
more about labels and metadata :ref:`here <environment-and-metadata>`.

``%help``
+++++++++

Any text in the ``%help`` section is transcribed into a metadata file in the
container during the build. This text can then be displayed using the
``run-help`` command.

Consider the example from the def file above:

.. code-block:: singularity

    %help
        This is a demo container used to illustrate a def file that uses all
        supported sections.

After building the help can be displayed like so:

.. code-block:: none

    $ singularity run-help my_container.sif
        This is a demo container used to illustrate a def file that uses all
        supported sections.

The `"Sections" part of the definition file documentation
<https://sylabs.io/guides/3.7/user-guide/definition_files.html#sections>`_
details all the sections and provides an example definition file that
makes use of all the sections.

Additional Singularity features
-------------------------------

Singularity has a wide range of features. You can find full details in
the `Singularity User Guide
<https://sylabs.io/guides/3.5/user-guide/index.html>`_ and we
highlight a couple of key features here that may be of use/interest:

**Remote Builder Capabilities:** If you have access to a platform with
Singularity installed but you don't have root access to create
containers, you may be able to use the `Remote
Builder <https://cloud.sylabs.io/builder>`_ functionality to offload the
process of building an image to remote cloud resources.  You'll need
to register for a **cloud token** via the link on the Remote Builder
page.

**Signing containers:** If you do want to share container image
(``.sif``) files directly with colleagues or collaborators, how can the
people you send an image to be sure that they have received the file
without it being tampered with or suffering from corruption during
transfer/storage? And how can you be sure that the same goes for any
container image file you receive from others? Singularity supports
signing containers. This allows a digital signature to be linked to
an image file. This signature can be used to verify that an image
file has been signed by the holder of a specific key and that the
file is unchanged from when it was signed. You can find full details
of how to use this functionality in the Singularity documentation on
`Signing and Verifying Containers <https://sylabs.io/guides/3.7/user-guide/signNverify.html>`_.
