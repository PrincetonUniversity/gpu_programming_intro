# NVIDIA CUDA Toolkit

![NVIDIA CUDA](https://upload.wikimedia.org/wikipedia/en/b/b9/Nvidia_CUDA_Logo.jpg)

The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) provides a comprehensive set of libraries and tools for developing and running GPU-accelerated applications.

List the available modules that are related to CUDA:

```
$ module avail cu

---------- /usr/local/share/Modules/modulefiles ----------
cudatoolkit/10.2  cudatoolkit/11.3  cudatoolkit/11.7       
cudatoolkit/11.1  cudatoolkit/11.4  cudnn/cuda-11.5/8.3.2
```

Run the following command to see which environment variables the `cudatoolkit` module is modifying:

```
$ module show cudatoolkit/11.7
-------------------------------------------------------------------
/usr/local/share/Modules/modulefiles/cudatoolkit/11.7:

module-whatis   {Sets up cudatoolkit117 11.7 in your environment}
prepend-path    PATH /usr/local/cuda-11.7/bin
prepend-path    LD_LIBRARY_PATH /usr/local/cuda-11.7/lib64
prepend-path    LIBRARY_PATH /usr/local/cuda-11.7/lib64
prepend-path    MANPATH /usr/local/cuda-11.7/doc/man
append-path     -d { } LDFLAGS -L/usr/local/cuda-11.7/lib64
append-path     -d { } INCLUDE -I/usr/local/cuda-11.7/include
append-path     CPATH /usr/local/cuda-11.7/include
append-path     -d { } FFLAGS -I/usr/local/cuda-11.7/include
append-path     -d { } LOCAL_LDFLAGS -L/usr/local/cuda-11.7/lib64
append-path     -d { } LOCAL_INCLUDE -I/usr/local/cuda-11.7/include
append-path     -d { } LOCAL_CFLAGS -I/usr/local/cuda-11.7/include
append-path     -d { } LOCAL_FFLAGS -I/usr/local/cuda-11.7/include
append-path     -d { } LOCAL_CXXFLAGS -I/usr/local/cuda-11.7/include
setenv          CUDA_HOME /usr/local/cuda-11.7
-------------------------------------------------------------------
```

Let's look at the files in /usr/local/cuda-11.7/bin:

```
$ ls -ltrh /usr/local/cuda-11.7/bin
total 86M
-rw-r--r--. 7 root root  417 May 29  2021 nvcc.profile
-rwxr-xr-x. 7 root root  285 May 29  2021 nvvp
-rwxr-xr-x. 1 root root  75K Apr  5  2022 cu++filt
-rwxr-xr-x. 5 root root 1.6K Apr  5  2022 nsight_ee_plugins_manage.sh
-rwxr-xr-x. 1 root root 107K Apr  5  2022 nvprune
-rwxr-xr-x. 1 root root 348K Apr  5  2022 cuda-memcheck
-rwxr-xr-x. 1 root root 304K Apr  5  2022 cuobjdump
-rwxr-xr-x. 1 root root  32M Apr  5  2022 nvdisasm
-rwxr-xr-x. 1 root root 5.5M Apr  5  2022 nvprof
-rwxr-xr-x. 2 root root  115 Apr  5  2022 compute-sanitizer
lrwxrwxrwx. 1 root root    4 Apr  5  2022 computeprof -> nvvp
-rwxr-xr-x. 1 root root 761K Apr  5  2022 cuda-gdbserver
-rwxr-xr-x. 1 root root  14M Apr  5  2022 cuda-gdb
-rwxr-xr-x. 1 root root 9.9M May  4 02:45 ptxas
-rwxr-xr-x. 1 root root  11M May  4 02:45 nvlink
-rwxr-xr-x. 1 root root 644K May  4 02:45 __nvcc_device_query
-rwxr-xr-x. 1 root root 6.3M May  4 02:45 nvcc
-rwxr-xr-x. 1 root root 272K May  4 02:45 fatbinary
-rwxr-xr-x. 1 root root 6.4M May  4 02:45 cudafe++
-rwxr-xr-x. 1 root root  83K May  4 02:45 bin2c
lrwxrwxrwx. 1 root root    3 May  4 11:41 nv-nsight-cu-cli -> ncu
lrwxrwxrwx. 1 root root    6 May  4 11:41 nv-nsight-cu -> ncu-ui
-rwxr-xr-x. 1 root root 2.8K May  4 11:41 ncu-ui
-rwxr-xr-x. 4 root root 3.0K May  4 11:41 ncu
-rwxr-xr-x. 1 root root  739 May  4 11:41 nsys-ui
-rwxr-xr-x. 4 root root  104 May  4 11:41 nsys-exporter
-rwxr-xr-x. 1 root root  751 May  4 11:41 nsys
-rwxr-xr-x. 3 root root   82 May  4 11:41 nsight-sys
drwxr-xr-x. 2 root root   43 Jun  6 10:13 crt
```

`nvcc` is the NVIDIA CUDA Compiler. Note that `nvcc` is built on `llvm` as [described here](https://developer.nvidia.com/cuda-llvm-compiler). To learn more about an executable, use the help option. For instance: `nvcc --help`.


Let's look at the libraries:

```
$ ls -lL /usr/local/cuda-11.7/lib64/lib*.so
-rwxr-xr-x. 1 root root   2120056 Apr  5  2022 /usr/local/cuda-11.7/lib64/libaccinj64.so
-rwxr-xr-x. 1 root root   1694872 Apr  5  2022 /usr/local/cuda-11.7/lib64/libcheckpoint.so
-rwxr-xr-x. 1 root root 348150584 Apr  5  2022 /usr/local/cuda-11.7/lib64/libcublasLt.so
-rwxr-xr-x. 1 root root 156720544 Apr  5  2022 /usr/local/cuda-11.7/lib64/libcublas.so
-rwxr-xr-x. 1 root root    671072 Apr 22 22:00 /usr/local/cuda-11.7/lib64/libcudart.so
-rwxr-xr-x. 1 root root 136837944 Apr  5  2022 /usr/local/cuda-11.7/lib64/libcufft.so
-rwxr-xr-x. 1 root root    773880 Apr  5  2022 /usr/local/cuda-11.7/lib64/libcufftw.so
-rwxr-xr-x. 1 root root     39296 Apr 28 14:39 /usr/local/cuda-11.7/lib64/libcufile_rdma.so
-rwxr-xr-x. 1 root root   1454840 Apr 28 14:39 /usr/local/cuda-11.7/lib64/libcufile.so
-rwxr-xr-x. 1 root root   2536416 Apr  5  2022 /usr/local/cuda-11.7/lib64/libcuinj64.so
-rwxr-xr-x. 1 root root   7091568 Apr  5  2022 /usr/local/cuda-11.7/lib64/libcupti.so
-rwxr-xr-x. 1 root root  87825840 Apr  5  2022 /usr/local/cuda-11.7/lib64/libcurand.so
-rwxr-xr-x. 1 root root 162685696 Apr  5  2022 /usr/local/cuda-11.7/lib64/libcusolverMg.so
-rwxr-xr-x. 1 root root 273555088 Apr  5  2022 /usr/local/cuda-11.7/lib64/libcusolver.so
-rwxr-xr-x. 1 root root 234731760 Apr  5  2022 /usr/local/cuda-11.7/lib64/libcusparse.so
-rwxr-xr-x. 1 root root   1610224 Apr 28 14:52 /usr/local/cuda-11.7/lib64/libnppc.so
-rwxr-xr-x. 1 root root  14279472 Apr 28 14:52 /usr/local/cuda-11.7/lib64/libnppial.so
-rwxr-xr-x. 1 root root   6382384 Apr 28 14:52 /usr/local/cuda-11.7/lib64/libnppicc.so
-rwxr-xr-x. 1 root root   9671832 Apr 28 14:52 /usr/local/cuda-11.7/lib64/libnppidei.so
-rwxr-xr-x. 1 root root  74745856 Apr 28 14:52 /usr/local/cuda-11.7/lib64/libnppif.so
-rwxr-xr-x. 1 root root  32179088 Apr 28 14:52 /usr/local/cuda-11.7/lib64/libnppig.so
-rwxr-xr-x. 1 root root   8385352 Apr 28 14:52 /usr/local/cuda-11.7/lib64/libnppim.so
-rwxr-xr-x. 1 root root  34034768 Apr 28 14:52 /usr/local/cuda-11.7/lib64/libnppist.so
-rwxr-xr-x. 1 root root    687528 Apr 28 14:52 /usr/local/cuda-11.7/lib64/libnppisu.so
-rwxr-xr-x. 1 root root   4485744 Apr 28 14:52 /usr/local/cuda-11.7/lib64/libnppitc.so
-rwxr-xr-x. 1 root root  17257720 Apr 28 14:52 /usr/local/cuda-11.7/lib64/libnpps.so
-rwxr-xr-x. 1 root root    724840 Apr  5  2022 /usr/local/cuda-11.7/lib64/libnvblas.so
-rwxr-xr-x. 1 root root   5554920 Apr  5  2022 /usr/local/cuda-11.7/lib64/libnvjpeg.so
-rwxr-xr-x. 1 root root  21654328 Apr  5  2022 /usr/local/cuda-11.7/lib64/libnvperf_host.so
-rwxr-xr-x. 1 root root   4498264 Apr  5  2022 /usr/local/cuda-11.7/lib64/libnvperf_target.so
-rwxr-xr-x. 1 root root   7075720 Apr  5  2022 /usr/local/cuda-11.7/lib64/libnvrtc-builtins.so
-rwxr-xr-x. 1 root root  45781472 Apr  5  2022 /usr/local/cuda-11.7/lib64/libnvrtc.so
-rwxr-xr-x. 3 root root     40136 Dec  8  2021 /usr/local/cuda-11.7/lib64/libnvToolsExt.so
-rwxr-xr-x. 3 root root     30856 Dec  8  2021 /usr/local/cuda-11.7/lib64/libOpenCL.so
-rwxr-xr-x. 1 root root    912728 Apr  5  2022 /usr/local/cuda-11.7/lib64/libpcsamplingutil.so
```

## cuDNN

There is also the [CUDA Deep Neural Net](https://developer.nvidia.com/cudnn) (cuDNN) library. It is external to the NVIDIA CUDA Toolkit and is used with TensorFlow, for instance, to provide GPU routines for training neural nets. See the available modules with:

```
$ module avail cudnn
```

## Conda Installations

When you install [CuPy](https://cupy.dev), for instance, Conda will include a CUDA Toolkit package (not the development version) and cuDNN:

<pre>
$ module load anaconda3/2022.5
$ conda create --name py-gpu cupy --channel conda-forge
...
  _libgcc_mutex      conda-forge/linux-64::_libgcc_mutex-0.1-conda_forge
  _openmp_mutex      conda-forge/linux-64::_openmp_mutex-4.5-2_gnu
  bzip2              conda-forge/linux-64::bzip2-1.0.8-h7f98852_4
  ca-certificates    conda-forge/linux-64::ca-certificates-2022.9.24-ha878542_0
  <b>cudatoolkit        conda-forge/linux-64::cudatoolkit-11.7.0-hd8887f6_10</b>
  cupy               conda-forge/linux-64::cupy-11.2.0-py310h9216885_0
  fastrlock          conda-forge/linux-64::fastrlock-0.8-py310hd8f1fbe_2
  ld_impl_linux-64   conda-forge/linux-64::ld_impl_linux-64-2.36.1-hea4e1c9_2
  libblas            conda-forge/linux-64::libblas-3.9.0-16_linux64_openblas
  libcblas           conda-forge/linux-64::libcblas-3.9.0-16_linux64_openblas
  libffi             conda-forge/linux-64::libffi-3.4.2-h7f98852_5
  libgcc-ng          conda-forge/linux-64::libgcc-ng-12.1.0-h8d9b700_16
  libgfortran-ng     conda-forge/linux-64::libgfortran-ng-12.1.0-h69a702a_16
  libgfortran5       conda-forge/linux-64::libgfortran5-12.1.0-hdcd56e2_16
  libgomp            conda-forge/linux-64::libgomp-12.1.0-h8d9b700_16
  liblapack          conda-forge/linux-64::liblapack-3.9.0-16_linux64_openblas
  libnsl             conda-forge/linux-64::libnsl-2.0.0-h7f98852_0
  libopenblas        conda-forge/linux-64::libopenblas-0.3.21-pthreads_h78a6416_3
  libsqlite          conda-forge/linux-64::libsqlite-3.39.4-h753d276_0
  libstdcxx-ng       conda-forge/linux-64::libstdcxx-ng-12.1.0-ha89aaad_16
  libuuid            conda-forge/linux-64::libuuid-2.32.1-h7f98852_1000
  libzlib            conda-forge/linux-64::libzlib-1.2.12-h166bdaf_4
  ncurses            conda-forge/linux-64::ncurses-6.3-h27087fc_1
  numpy              conda-forge/linux-64::numpy-1.23.3-py310h53a5b5f_0
  openssl            conda-forge/linux-64::openssl-3.0.5-h166bdaf_2
  pip                conda-forge/noarch::pip-22.2.2-pyhd8ed1ab_0
  python             conda-forge/linux-64::python-3.10.6-ha86cf86_0_cpython
  python_abi         conda-forge/linux-64::python_abi-3.10-2_cp310
  readline           conda-forge/linux-64::readline-8.1.2-h0f457ee_0
  setuptools         conda-forge/noarch::setuptools-65.4.1-pyhd8ed1ab_0
  tk                 conda-forge/linux-64::tk-8.6.12-h27826a3_0
  tzdata             conda-forge/noarch::tzdata-2022d-h191b570_0
  wheel              conda-forge/noarch::wheel-0.37.1-pyhd8ed1ab_0
  xz                 conda-forge/linux-64::xz-5.2.6-h166bdaf_0
</pre>
