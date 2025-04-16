.. _namespc-cgroup:

Namespaces and cgroups
======================

What Are Namespaces?
____________________

Namespaces have been part of the Linux kernel since about 2002, and over time more
tooling and namespace types have been added. Real container support was added to
the Linux kernel only in 2013, however. This is what made namespaces really useful
and brought them to the masses.

.. callout :: Definition according to `Wikipedia <https://en.wikipedia.org/wiki/Linux_namespaces>`__

  `Namespaces are a feature of the Linux kernel that partitions kernel resources
  such that one set of processes sees one set of resources while another set of
  processes sees a different set of resources.`

In other words, the key feature of namespaces is that they isolate processes from
each other. On a server where you are running many different services, isolating
each service and its associated processes from other services means that there is
a smaller blast radius for changes, as well as a smaller footprint for security‑related
concerns.

Using containers during the development process gives the developer an isolated
environment that looks and feels like a complete VM. It’s not a VM, though – it’s
a process running on a server somewhere. If the developer starts two containers,
there are two processes running on a single server somewhere – but they are isolated
from each other.

Types of Namespaces
+++++++++++++++++++

Within the Linux kernel, there are different types of namespaces. Each namespace
has its own unique properties:

- A `user namespace <https://man7.org/linux/man-pages/man7/user_namespaces.7.html>`_
  has its own set of user IDs and group IDs for assignment to processes. In particular,
  this means that a process can have root privilege within its user namespace without
  having it in other user namespaces.

- A `process ID (PID) namespace <https://man7.org/linux/man-pages/man7/pid_namespaces.7.html>`_
  assigns a set of PIDs to processes that are independent from the set of PIDs in other namespaces.
  The first process created in a new namespace has PID 1 and child processes are assigned subsequent PIDs.
  If a child process is created with its own PID namespace, it has PID 1 in that namespace
  as well as its PID in the parent process’ namespace. See below for an example.

- A `network namespace <https://man7.org/linux/man-pages/man7/network_namespaces.7.html>`_
  has an independent network stack: its own private routing table, set of IP addresses,
  socket listing, connection tracking table, firewall, and other network‑related resources.

- A `mount namespace <https://man7.org/linux/man-pages/man7/mount_namespaces.7.html>`_
  has an independent list of mount points seen by the processes in the namespace. This means
  that you can mount and unmount filesystems in a mount namespace without affecting the host filesystem.

- An `interprocess communication (IPC) namespace <https://man7.org/linux/man-pages/man7/ipc_namespaces.7.html>`_
  has its own IPC resources.

- A `UNIX Time‑Sharing (UTS) namespace <https://man7.org/linux/man-pages/man7/uts_namespaces.7.html>`_
  allows a single system to appear to have different host and domain names to different processes.

An Example of Parent and Child PID Namespaces
+++++++++++++++++++++++++++++++++++++++++++++

In the diagram below, there are three PID namespaces – a parent namespace and
two child namespaces. Within the parent namespace, there are four processes,
named PID1 through PID4. These are normal processes which can all see each
other and share resources.

The child processes with PID2 and PID3 in the parent namespace also belong to
their own PID namespaces in which their PID is 1. From within a child namespace,
the PID1 process cannot see anything outside. For example, PID1 in both child
namespaces cannot see PID4 in the parent namespace.

This provides isolation between (in this case) processes within different namespaces.

.. figure :: https://www.nginx.com/wp-content/uploads/2021/07/Namespaces-cgroups_PID-namespaces.svg

  `(Image Source) <https://www.nginx.com/blog/what-are-namespaces-cgroups-how-do-they-work>`_

Creating a Namespace
++++++++++++++++++++

With all that theory under our belts, let’s cement our understanding by actually
creating a new namespace. The Linux `unshare <https://man7.org/linux/man-pages/man1/unshare.1.html>`_
command is a good place to start. The manual page indicates that it does exactly what we want:

.. code-block :: bash

  NAME
          unshare - run program in new name namespaces

Let's see our user ID, group, and so on. While in PWD we have root privileges,
to use this command we don't have to have root privileges:

.. code-block :: bash

  $ id
  uid=0(root) gid=0(root) groups=0(root),1(bin),2(daemon),3(sys),4(adm),6(disk),10(wheel),11(floppy),20(dialout),26(tape),27(video)

Now we run the following unshare command to create a new namespace with its own
user and PID namespaces. We map the root user to the new namespace (in other words,
we have root privilege within the new namespace), mount a new proc filesystem,
and fork my process (in this case, bash) in the newly created namespace.

.. code-block :: bash

  unshare --user --pid --map-root-user --mount-proc --fork bash

The command above accomplishes the same thing as issuing
the ``<runtime> exec -it <image> /bin/bash`` command in a running container.

.. callout :: ``ps`` (process status) command in PWD

  The avaiable ``ps`` command in PWD doesn't show the output in a desired state.
  We need to install it manually using

  .. code-block :: bash

    apk add --no-cache procps

The ``ps -ef`` command shows there are two processes running – **bash** and
the **ps** command itself – and the id command confirms that I’m **root** in the new
namespace (which is also indicated by the changed command prompt):

.. code-block ::

  $ ps -ef
  UID        PID  PPID  C STIME TTY          TIME CMD
  root         1     0  0 15:46 pts/1    00:00:00 bash
  root        21     1  0 15:56 pts/1    00:00:00 ps -ef

  $ id
  uid=0(root) gid=0(root) groups=0(root),65534(nobody),65534(nobody)

The crucial thing to notice is that I can see only the two processes in my namespace,
not any other processes running on the system. I am completely isolated within my own namespace.

(** Above exercise can be also done on the Vega.)

Looking at a Namespace from the Outside
+++++++++++++++++++++++++++++++++++++++

Although we can’t see other processes from within the namespace, with the lsns (list namespaces)
command we can list all available namespaces and display information about them,
from the perspective of the parent namespace (outside the new namespace).

The output shows the namespaces – of types user, mnt, and pid – which correspond
to the arguments on the unshare command we ran above. From this external perspective,
each namespace is running as user $USER, not root, whereas inside the namespace processes run as root,
with access to all of the expected resources.

.. code-block :: bash

  $ lsns --output-all
          NS TYPE   PATH              NPROCS PID PPID COMMAND UID USER    NETNSID NSFS
  4026531835 cgroup /proc/1/ns/cgroup      2   1    0 bash      0 root
  4026533087 uts    /proc/1/ns/uts         2   1    0 bash      0 root
  4026533090 ipc    /proc/1/ns/ipc         2   1    0 bash      0 root
  4026533093 net    /proc/1/ns/net         2   1    0 bash      0 root unassigned
  4026537060 pid    /proc/1/ns/pid         2   1    0 bash      0 root
  4026537071 user   /proc/1/ns/user        2   1    0 bash      0 root
  4026537072 mnt    /proc/1/ns/mnt         2   1    0 bash      0 root

Namespaces and Containers
+++++++++++++++++++++++++

Namespaces are one of the technologies that containers are built on, used to enforce
segregation of resources. We’ve shown how to create namespaces manually, but container
runtimes like Docker makes things easier by creating namespaces on your behalf.

What Are cgroups?
_________________
A control group (cgroup) is a Linux kernel feature that limits, accounts for,
and isolates the resource usage (CPU, memory, disk I/O, network, and so on) of a collection of processes.

Cgroups provide the following features:

- **Resource limits**: You can configure a cgroup to limit how much of a particular
  resource (memory or CPU, for example) a process can use.

- **Prioritization**: You can control how much of a resource (CPU, disk, or network)
  a process can use compared to processes in another cgroup when there is resource contention.

- **Accounting**: Resource limits are monitored and reported at the cgroup level.

- **Control**: You can change the status (frozen, stopped, or restarted) of all
  processes in a cgroup with a single command.

So basically you use cgroups to control how much of a given key resource (CPU, memory, network, and disk I/O)
can be accessed or used by a process or set of processes. Cgroups are a key component
of containers because there are often multiple processes running in a container
that you need to control together. In a Kubernetes environment, cgroups can be
used to implement resource requests and limits and corresponding QoS classes at the pod level.

The following diagram illustrates how when you allocate a particular percentage
of available system resources to a cgroup (in this case **cgroup‑1**),
he remaining percentage is available to other cgroups (and individual processes) on the system.

Cgroup Versions
+++++++++++++++

According to `Wikipedia <https://en.wikipedia.org/wiki/Cgroups>`__, the first version
of cgroups was merged into the Linux kernel mainline in late 2007 or early 2008,
and “the documentation of cgroups‑v2 first appeared in [the] Linux kernel … [in] 2016”.
Among the many changes in version 2, the big ones are a much simplified tree architecture,
new features and interfaces in the cgroup hierarchy, and better
accommodation of “rootless” containers (with non‑zero UIDs).

.. figure :: https://www.nginx.com/wp-content/uploads/2021/07/Namespaces-cgroups_resource-limits.svg


  `(Image Source) <https://www.nginx.com/blog/what-are-namespaces-cgroups-how-do-they-work>`_

`(Image Source) <https://www.nginx.com/blog/what-are-namespaces-cgroups-how-do-they-work>`_

Creating a cgroup
+++++++++++++++++

The following command creates a v1 cgroup (you can tell by pathname format)
called foo and sets the memory limit for it to 50,000,000 bytes (50 MB).

.. code-block :: bash

  $ mkdir -p /sys/fs/cgroup/memory/foo
  $ sudo echo 50000000 > /sys/fs/cgroup/memory/foo/memory.limit_in_bytes

If we know check the mem limits, we get

.. code-block :: bash

  sudo cat /sys/fs/cgroup/memory/foo/memory.limit_in_bytes
  49999872

Now, let's create a test bash file to check cgroup functionality. A simple example
of such shell is:

.. code-block :: bash

  $ vim test.sh

  #!/bin/sh
  while [ 1 ]; do
      echo "hellp world"
      sleep 60
  done

``test.sh`` is a shell script, which prints a message to the screen
and then sleeps for 60 seconds. It is fine for our purposes because it is in
an infinite loop.

.. code-block :: bash

  $ sh ./test.sh &
  [1] 31344
  hello world

``test.sh`` is started in the background and its PID is reported as 31344.
The script produces its output and then we assign the process to the cgroup
by piping its PID into the cgroup file ``/sys/fs/cgroup/memory/foo/cgroup.procs``.

.. code-block :: bash

  $ sudo echo 31344 > /sys/fs/cgroup/memory/foo/cgroup.procs

To validate that my process is in fact subject to the memory limits that we defined
for cgroup foo, we run the following ps command. The -o cgroup flag displays
the cgroups to which the specified process (31344) belongs. The output confirms
that its memory cgroup is foo.

.. code-block :: bash

  $ ps -o cgroup 31344
  CGROUP
  11:memory:/docker/874edaaa7ef8e61e283b438077e82c3435e53c5bedc91ba63ea84eca0993678f/foo,10:blkio:/docker/874eda

We can also check the amount of memory currently ``test.sh`` is using with the command below.

.. code-block :: bash

  $ sudo cat /sys/fs/cgroup/memory/foo/memory.usage_in_bytes
  1712128

Namespaces and cgroups are the building blocks for containers and modern applications.
Having an understanding of how they work is important as we refactor applications
to more modern architectures.

Namespaces provide isolation of system resources, and cgroups allow for fine‑grained
control and enforcement of limits for those resources.

Containers are not the only way that you can use namespaces and cgroups.
Namespaces and cgroup interfaces are built into the Linux kernel, which means that other applications can use them to provide separation and resource constraints.
