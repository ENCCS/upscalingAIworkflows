.. _create_contain:

Creating your own container images
==================================

There are lots of reasons why you might want to create your **own**
Docker image.

- You can't find a container with all the tools you need on Docker
  Hub.
- You want to have a container to "archive" all the specific software
  versions you ran for a project
- You want to share your workflow with someone else.

Interactive installation
------------------------

Before creating a reproducible installation, let's experiment with
installing software inside a container. Start the `alpine` container
from before, interactively:

.. code-block:: bash

    docker run -it ubuntu sh

Because this is a basic container, there's a lot of things not
installed -- for example, `python3`.

.. code-block:: bash

   /# python3

Output

.. code-block:: text

   sh: 1: python3: not found

Inside the container, we can run commands to install Python 3. The
Ubuntu version of Linux has a installation tool called `apt install` that we
can use to install Python 3.

.. code-block:: bash

   /# apt update && apt install -y python3 pip

We can test our installation by running a Python command:

.. code-block:: docker

   /# python3 --version


Once Python is installed, we can add Python packages using the pip
package installer:

.. code-block:: bash

   /# pip install cython


Once we exit, these changes are not saved to a new container by
default. There is a command that will "snapshot" our changes, but
building containers this way is not very reproducible. Instead, we're
going to take what we've learned from this interactive installation
and create our container from a reproducible recipe, known as a
`Dockerfile`.

If you haven't already, exit out of the interactively running container.

.. code-block:: bash

   /# exit

Put installation instructions in a `Dockerfile`
-----------------------------------------------

A `Dockerfile` is a plain text file with keywords and commands that
can be used to create a new container image.

Every Dockerfile is composed of three main parts as shown below.

.. code-block:: dockerfile

   FROM <EXISTING IMAGE>
   RUN <INSTALL CMDS FROM SHELL>
   RUN <INSTALL CMDS FROM SHELL>
   CMD <CMD TO RUN BY DEFAULT>

Let's break this file down:

- The first line, `FROM`, indicates which container we're starting with.
- The next two lines `RUN`, will indicate installation commands we
  want to run. These are the same commands that we used interactively
  above.
- The last line, `CMD` indicates the default command we want the
  container to run, if no other command is provided.

.. exercise:: Take a Guess

   Do you have any ideas about what we should use to fill in the
   sample Dockerfile to replicate the installation we did above?

   .. solution::

      Based on our experience above, edit the `Dockerfile` (in your
      text editor of choice) to look like this:

      .. code-block:: dockerfile

         FROM ubuntu:18.04
         RUN apt update && apt install -y python3 pip
         RUN pip install cython
         CMD python3 --version

The recipe provided by this Dockerfile will use Ubuntu Linux as the
base container, add Python and the Cython library, and set a default
print command.

Create a new Docker image
-------------------------

So far, we just have a file. We want Docker to take this file, run the
install commands inside, and then save the resulting container as a
new container image. To do this we will use the `docker build`
command.

We have to provide `docker build` with two pieces of information:

- the location of the `Dockerfile`
- the name of the new image. Remember the naming scheme from before?
  You should name your new image with your Docker Hub username and a
  name for the container, like this: ``USERNAME/CONTAINERNAME``

All together, the build command will look like this:

.. code-block:: bash

   docker build -t USERNAME/CONTAINERNAME .


The `-t` option names the container; the final dot indicates that the
`Dockerfile` is in our current directory.

For example, if my user name was `alice` and I wanted to call my
image `alpine-python`, I would use this command:

.. code-block:: bash

   docker build -t alice/ubuntu-python .

.. exercise:: Review!

   1. Think back to earlier. What command can you run to check if
      your image was created successfully? (Hint: what command shows
      the images on your computer?)

   2. We didn't specify a tag for our image name. What did
      Docker automatically use?

   3. What command will run the container you've created? What
      should happen by default if you run the container? Can you make
      it do something different, like print "hello world"?

   .. solution::

      1. To see your new image, run `docker image ls`. You should
	 see the name of your new image under the "REPOSITORY" heading.

      2. In the output of `docker image ls`, you can see that
	 Docker has automatically used the `latest` tag for our new
	 image.

      3. We want to use `docker run` to run the container.

	 .. code-block:: bash

	    docker run alice/ubuntu-python

	 should run the container and print out our default
	 message, including the version of Linux and Python.

	 .. code-block:: bash

	   docker run alice/ubuntu-python echo "Hello World"

         will run the container and print out "Hello world" instead.


While it may not look like you have achieved much, you have already
effected the combination of a lightweight Linux operating system with
your specification to run a given command that can operate reliably on
macOS, Microsoft Windows, Linux and on the cloud!


Share your new container on Docker Hub
--------------------------------------

Images that you release publicly can be stored on the Docker Hub for
free.  If you name your image as described above, with your Docker Hub
username, all you need to do is run the opposite of `docker pull` --
`docker push`.

.. code-block:: bash

   docker push alice/ubuntu-python

Make sure to substitute the full name of your container!

In a web browser, open <https://hub.docker.com>, and on your user page
you should now see your container listed, for anyone to use or build
on.

.. callout:: Logging In

   Technically, you have to be logged into Docker on your computer for
   this to work.  Usually it happens by default, but if `docker push`
   doesn't work for you, run `docker login` first, enter your Docker
   Hub username and password, and then try `docker push` again.

You can rename images using the `docker tag` command. For example,
imagine someone named Alice has been working on a workflow container
and called it `workflow-test` on her own computer. She now wants to
share it in her `alice` Docker Hub account with the name
`workflow-complete` and a tag of `v1`. Her `docker tag` command would
look like this:

.. code-block:: bash

   docker tag workflow-test alice/workflow-complete:v1


She could then push the re-named container to Docker Hub, using
`docker push alice/workflow-complete:v1`