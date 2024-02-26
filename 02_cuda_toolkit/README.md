# NVIDIA CUDA Toolkit

![NVIDIA CUDA](https://upload.wikimedia.org/wikipedia/en/b/b9/Nvidia_CUDA_Logo.jpg)

The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) provides a comprehensive set of libraries and tools for developing and running GPU-accelerated applications.

List the available modules that are related to CUDA:

```
$ module avail cudatoolkit
-------------------- /usr/local/share/Modules/modulefiles --------------------
cudatoolkit/10.2  cudatoolkit/11.3  cudatoolkit/11.7  cudatoolkit/12.2
cudatoolkit/11.1  cudatoolkit/11.4  cudatoolkit/12.0  cudatoolkit/12.3
```

Run the following command to see which environment variables the `cudatoolkit` module is modifying:

```
$ module show cudatoolkit/12.2
-------------------------------------------------------------------
/usr/local/share/Modules/modulefiles/cudatoolkit/12.2:

module-whatis   {Sets up cudatoolkit122 12.2 in your environment}
prepend-path    PATH /usr/local/cuda-12.2/bin
prepend-path    LD_LIBRARY_PATH /usr/local/cuda-12.2/lib64
prepend-path    LIBRARY_PATH /usr/local/cuda-12.2/lib64
prepend-path    MANPATH /usr/local/cuda-12.2/doc/man
append-path     -d { } LDFLAGS -L/usr/local/cuda-12.2/lib64
append-path     -d { } INCLUDE -I/usr/local/cuda-12.2/include
append-path     CPATH /usr/local/cuda-12.2/include
append-path     -d { } FFLAGS -I/usr/local/cuda-12.2/include
append-path     -d { } LOCAL_LDFLAGS -L/usr/local/cuda-12.2/lib64
append-path     -d { } LOCAL_INCLUDE -I/usr/local/cuda-12.2/include
append-path     -d { } LOCAL_CFLAGS -I/usr/local/cuda-12.2/include
append-path     -d { } LOCAL_FFLAGS -I/usr/local/cuda-12.2/include
append-path     -d { } LOCAL_CXXFLAGS -I/usr/local/cuda-12.2/include
setenv          CUDA_HOME /usr/local/cuda-12.2
-------------------------------------------------------------------
```

Let's look at the files in `/usr/local/cuda-12.2/bin`:

```
$ ls -ltrh /usr/local/cuda-12.2/bin
total 148M
-rw-r--r--. 9 root root  417 May 29  2021 nvcc.profile
-rwxr-xr-x. 9 root root  285 May 29  2021 nvvp
-rwxr-xr-x. 7 root root 1.6K Apr 30 22:21 nsight_ee_plugins_manage.sh
-rwxr-xr-x. 1 root root  49M Apr 30 22:29 nvdisasm
-rwxr-xr-x. 1 root root 516K Apr 30 22:30 cuobjdump
-rwxr-xr-x. 1 root root 107K Apr 30 22:34 nvprune
-rwxr-xr-x. 1 root root  75K Apr 30 22:37 cu++filt
-rwxr-xr-x. 1 root root 789K Apr 30 23:50 cuda-gdbserver
-rwxr-xr-x. 1 root root  15M Apr 30 23:50 cuda-gdb
-rwxr-xr-x. 1 root root  112 May  1 02:54 compute-sanitizer
lrwxrwxrwx. 1 root root    4 May  9 01:30 computeprof -> nvvp
-rwxr-xr-x. 1 root root 5.8M May  9 01:53 nvprof
-rwxr-xr-x. 1 root root  25M Jun 13 23:01 ptxas
-rwxr-xr-x. 1 root root  25M Jun 13 23:01 nvlink
-rwxr-xr-x. 1 root root  11K Jun 13 23:01 __nvcc_device_query
-rwxr-xr-x. 1 root root  21M Jun 13 23:01 nvcc
-rwxr-xr-x. 1 root root 280K Jun 13 23:01 fatbinary
-rwxr-xr-x. 1 root root 6.6M Jun 13 23:01 cudafe++
-rwxr-xr-x. 1 root root  83K Jun 13 23:01 bin2c
-rwxr-xr-x. 1 root root 3.6K Jun 26 03:15 ncu-ui
-rwxr-xr-x. 1 root root 3.8K Jun 26 03:15 ncu
-rwxr-xr-x. 1 root root  847 Jun 26 03:15 nsys-ui
-rwxr-xr-x. 6 root root  104 Jun 26 03:15 nsys-exporter
-rwxr-xr-x. 1 root root  751 Jun 26 03:15 nsys
-rwxr-xr-x. 1 root root  209 Jun 26 03:15 nsight-sys
drwxr-xr-x. 2 root root   43 Aug 31 08:31 crt
```

`nvcc` is the NVIDIA CUDA Compiler. Note that `nvcc` is built on `llvm` as [described here](https://developer.nvidia.com/cuda-llvm-compiler). To learn more about an executable, use the help option. For instance: `nvcc --help`.


Let's look at the libraries:

```
$ ls -lL /usr/local/cuda-12.2/lib64/lib*.so
-rwxr-xr-x. 1 root root   2412216 May  9 01:53 /usr/local/cuda-12.2/lib64/libaccinj64.so
-rwxr-xr-x. 1 root root   1538200 May  9 01:43 /usr/local/cuda-12.2/lib64/libcheckpoint.so
-rwxr-xr-x. 1 root root 515090264 Apr 30 22:44 /usr/local/cuda-12.2/lib64/libcublasLt.so
-rwxr-xr-x. 1 root root 107473968 Apr 30 22:44 /usr/local/cuda-12.2/lib64/libcublas.so
-rwxr-xr-x. 1 root root    687456 Apr 30 22:31 /usr/local/cuda-12.2/lib64/libcudart.so
-rwxr-xr-x. 1 root root 178391592 Apr 30 22:47 /usr/local/cuda-12.2/lib64/libcufft.so
-rwxr-xr-x. 1 root root   1626632 Apr 30 22:47 /usr/local/cuda-12.2/lib64/libcufftw.so
-rwxr-xr-x. 2 root root     43320 May 17 13:38 /usr/local/cuda-12.2/lib64/libcufile_rdma.so
-rwxr-xr-x. 1 root root   2957904 May 17 13:38 /usr/local/cuda-12.2/lib64/libcufile.so
-rwxr-xr-x. 1 root root   2832640 May  9 01:53 /usr/local/cuda-12.2/lib64/libcuinj64.so
-rw-r--r--. 3 root root   7501840 May 24 22:40 /usr/local/cuda-12.2/lib64/libcupti.so
-rwxr-xr-x. 1 root root  96857520 Apr 30 22:38 /usr/local/cuda-12.2/lib64/libcurand.so
-rwxr-xr-x. 1 root root  81995920 Apr 30 23:19 /usr/local/cuda-12.2/lib64/libcusolverMg.so
-rwxr-xr-x. 1 root root 115329376 Apr 30 23:19 /usr/local/cuda-12.2/lib64/libcusolver.so
-rwxr-xr-x. 1 root root 263461056 Apr 30 22:58 /usr/local/cuda-12.2/lib64/libcusparse.so
-rwxr-xr-x. 1 root root   1626608 Apr 30 23:03 /usr/local/cuda-12.2/lib64/libnppc.so
-rwxr-xr-x. 1 root root  16315184 Apr 30 23:03 /usr/local/cuda-12.2/lib64/libnppial.so
-rwxr-xr-x. 1 root root   6763312 Apr 30 23:03 /usr/local/cuda-12.2/lib64/libnppicc.so
-rwxr-xr-x. 1 root root  10634392 Apr 30 23:03 /usr/local/cuda-12.2/lib64/libnppidei.so
-rwxr-xr-x. 1 root root  95934568 Apr 30 23:03 /usr/local/cuda-12.2/lib64/libnppif.so
-rwxr-xr-x. 1 root root  38855568 Apr 30 23:03 /usr/local/cuda-12.2/lib64/libnppig.so
-rwxr-xr-x. 1 root root   9261896 Apr 30 23:03 /usr/local/cuda-12.2/lib64/libnppim.so
-rwxr-xr-x. 1 root root  38048848 Apr 30 23:03 /usr/local/cuda-12.2/lib64/libnppist.so
-rwxr-xr-x. 1 root root    699816 Apr 30 23:03 /usr/local/cuda-12.2/lib64/libnppisu.so
-rwxr-xr-x. 1 root root   5390960 Apr 30 23:03 /usr/local/cuda-12.2/lib64/libnppitc.so
-rwxr-xr-x. 1 root root  19191032 Apr 30 23:03 /usr/local/cuda-12.2/lib64/libnpps.so
-rwxr-xr-x. 1 root root    737048 Apr 30 22:44 /usr/local/cuda-12.2/lib64/libnvblas.so
-rwxr-xr-x. 1 root root  49912352 Jun 14 00:41 /usr/local/cuda-12.2/lib64/libnvJitLink.so
-rwxr-xr-x. 1 root root   6607664 Apr 30 22:33 /usr/local/cuda-12.2/lib64/libnvjpeg.so
-rwxr-xr-x. 1 root root  28436024 May  9 01:43 /usr/local/cuda-12.2/lib64/libnvperf_host.so
-rwxr-xr-x. 1 root root   5977336 May  9 01:43 /usr/local/cuda-12.2/lib64/libnvperf_target.so
-rwxr-xr-x. 1 root root   2434952 Jun 14 00:46 /usr/local/cuda-12.2/lib64/libnvrtc-builtins.so
-rwxr-xr-x. 1 root root  58373448 Jun 14 00:46 /usr/local/cuda-12.2/lib64/libnvrtc.so
-rwxr-xr-x. 5 root root     40136 Dec  8  2021 /usr/local/cuda-12.2/lib64/libnvToolsExt.so
-rwxr-xr-x. 5 root root     30856 Dec  8  2021 /usr/local/cuda-12.2/lib64/libOpenCL.so
-rwxr-xr-x. 2 root root    912728 May  9 01:43 /usr/local/cuda-12.2/lib64/libpcsamplingutil.so
```

## cuDNN

There is also the [CUDA Deep Neural Net](https://developer.nvidia.com/cudnn) (cuDNN) library. It is external to the NVIDIA CUDA Toolkit and is used with TensorFlow, for instance, to provide GPU routines for training neural nets. See the available modules with:

```
$ module avail cudnn
```

## Conda Installations

When you install [CuPy](https://cupy.dev), for instance, which is like NumPy for GPUs, Conda will include the CUDA libraries:

<pre>
$ module load anaconda3/2023.9
$ conda create --name cupy-env cupy --channel conda-forge
...
   _libgcc_mutex      conda-forge/linux-64::_libgcc_mutex-0.1-conda_forge
  _openmp_mutex      conda-forge/linux-64::_openmp_mutex-4.5-2_gnu
  bzip2              conda-forge/linux-64::bzip2-1.0.8-h7f98852_4
  ca-certificates    conda-forge/linux-64::ca-certificates-2023.7.22-hbcca054_0
  cuda-cudart        conda-forge/linux-64::cuda-cudart-12.0.107-h59595ed_6
  cuda-cudart_linux~ conda-forge/noarch::cuda-cudart_linux-64-12.0.107-h59595ed_6
  cuda-nvrtc         conda-forge/linux-64::cuda-nvrtc-12.0.76-h59595ed_1
  cuda-nvtx          conda-forge/linux-64::cuda-nvtx-12.0.76-hcb278e6_0
  cuda-version       conda-forge/noarch::cuda-version-12.0-hffde075_2
  cupy               conda-forge/linux-64::cupy-12.2.0-py311h412bc61_3
  fastrlock          conda-forge/linux-64::fastrlock-0.8.2-py311hb755f60_1
  ld_impl_linux-64   conda-forge/linux-64::ld_impl_linux-64-2.40-h41732ed_0
  libblas            conda-forge/linux-64::libblas-3.9.0-18_linux64_openblas
  libcblas           conda-forge/linux-64::libcblas-3.9.0-18_linux64_openblas
  <b>libcublas          conda-forge/linux-64::libcublas-12.0.1.189-hcb278e6_2</b>
  <b>libcufft           conda-forge/linux-64::libcufft-11.0.0.21-hcb278e6_1</b>
  <b>libcurand          conda-forge/linux-64::libcurand-10.3.1.50-hcb278e6_0</b>
  <b>libcusolver        conda-forge/linux-64::libcusolver-11.4.2.57-hcb278e6_1</b>
  <b>libcusparse        conda-forge/linux-64::libcusparse-12.0.0.76-hcb278e6_1</b>
  libexpat           conda-forge/linux-64::libexpat-2.5.0-hcb278e6_1
  libffi             conda-forge/linux-64::libffi-3.4.2-h7f98852_5
  libgcc-ng          conda-forge/linux-64::libgcc-ng-13.2.0-h807b86a_2
  libgfortran-ng     conda-forge/linux-64::libgfortran-ng-13.2.0-h69a702a_2
  libgfortran5       conda-forge/linux-64::libgfortran5-13.2.0-ha4646dd_2
  libgomp            conda-forge/linux-64::libgomp-13.2.0-h807b86a_2
  liblapack          conda-forge/linux-64::liblapack-3.9.0-18_linux64_openblas
  libnsl             conda-forge/linux-64::libnsl-2.0.0-hd590300_1
  libnvjitlink       conda-forge/linux-64::libnvjitlink-12.0.76-hcb278e6_1
  libopenblas        conda-forge/linux-64::libopenblas-0.3.24-pthreads_h413a1c8_0
  libsqlite          conda-forge/linux-64::libsqlite-3.43.0-h2797004_0
  libstdcxx-ng       conda-forge/linux-64::libstdcxx-ng-13.2.0-h7e041cc_2
  libuuid            conda-forge/linux-64::libuuid-2.38.1-h0b41bf4_0
  libzlib            conda-forge/linux-64::libzlib-1.2.13-hd590300_5
  ncurses            conda-forge/linux-64::ncurses-6.4-hcb278e6_0
  numpy              conda-forge/linux-64::numpy-1.26.0-py311h64a7726_0
  openssl            conda-forge/linux-64::openssl-3.1.3-hd590300_0
  pip                conda-forge/noarch::pip-23.2.1-pyhd8ed1ab_0
  python             conda-forge/linux-64::python-3.11.6-hab00c5b_0_cpython
  python_abi         conda-forge/linux-64::python_abi-3.11-4_cp311
  readline           conda-forge/linux-64::readline-8.2-h8228510_1
  setuptools         conda-forge/noarch::setuptools-68.2.2-pyhd8ed1ab_0
  tk                 conda-forge/linux-64::tk-8.6.13-h2797004_0
  tzdata             conda-forge/noarch::tzdata-2023c-h71feb2d_0
  wheel              conda-forge/noarch::wheel-0.41.2-pyhd8ed1ab_0
  xz                 conda-forge/linux-64::xz-5.2.6-h166bdaf_0
</pre>

When using `pip` to do the installation, one needs to load the `cudatoolkit` module since that dependency is assumed to be available on the local system. The Conda approach installs all the dependencies so one does not load the module.
