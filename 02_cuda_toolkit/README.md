# NVIDIA CUDA Toolkit

![NVIDIA CUDA](https://upload.wikimedia.org/wikipedia/en/b/b9/Nvidia_CUDA_Logo.jpg)

The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) provides a comprehensive set of libraries and tools for developing and running GPU-accelerated applications.

List the available modules that are related to CUDA:

```
$ module avail cu

-------------------------- /usr/local/share/Modules/modulefiles ---------------------------
cudatoolkit/10.2  cudatoolkit/11.1  cudatoolkit/11.3  cudatoolkit/11.4
```

Run the following command to see which environment variables the `cudatoolkit` module is modifying:

```
$ module show cudatoolkit/11.4
-------------------------------------------------------------------
/usr/local/share/Modules/modulefiles/cudatoolkit/11.4:

module-whatis   {Sets up cudatoolkit114 11.4 in your environment}
prepend-path    PATH /usr/local/cuda-11.4/bin
prepend-path    LD_LIBRARY_PATH /usr/local/cuda-11.4/lib64
prepend-path    LIBRARY_PATH /usr/local/cuda-11.4/lib64
prepend-path    MANPATH /usr/local/cuda-11.4/doc/man
append-path     -d { } LDFLAGS -L/usr/local/cuda-11.4/lib64
append-path     -d { } INCLUDE -I/usr/local/cuda-11.4/include
append-path     CPATH /usr/local/cuda-11.4/include
append-path     -d { } FFLAGS -I/usr/local/cuda-11.4/include
append-path     -d { } LOCAL_LDFLAGS -L/usr/local/cuda-11.4/lib64
append-path     -d { } LOCAL_INCLUDE -I/usr/local/cuda-11.4/include
append-path     -d { } LOCAL_CFLAGS -I/usr/local/cuda-11.4/include
append-path     -d { } LOCAL_FFLAGS -I/usr/local/cuda-11.4/include
append-path     -d { } LOCAL_CXXFLAGS -I/usr/local/cuda-11.4/include
-------------------------------------------------------------------
```

Let's look at the files in /usr/local/cuda-11.4/bin:

```
$ ls -ltrh /usr/local/cuda-11.4/bin

-rwxr-xr-x. 1 root root 232K May 27 22:56 cuobjdump
-rw-r--r--. 5 root root  417 May 29 18:46 nvcc.profile
-rwxr-xr-x. 5 root root  285 May 29 18:46 nvvp
-rwxr-xr-x. 1 root root 348K Jul 14 23:30 cuda-memcheck
-rwxr-xr-x. 1 root root  33M Jul 14 23:30 nvdisasm
-rwxr-xr-x. 1 root root 760K Jul 14 23:32 cuda-gdbserver
-rwxr-xr-x. 1 root root  13M Jul 14 23:32 cuda-gdb
-rwxr-xr-x. 1 root root  75K Jul 14 23:39 cu++filt
-rwxr-xr-x. 1 root root  99K Jul 14 23:39 nvprune
-rwxr-xr-x. 4 root root 1.6K Jul 14 23:46 nsight_ee_plugins_manage.sh
lrwxrwxrwx. 1 root root    4 Jul 15 14:04 computeprof -> nvvp
-rwxr-xr-x. 1 root root 9.5M Jul 15 14:06 ptxas
-rwxr-xr-x. 1 root root 9.7M Jul 15 14:06 nvlink
-rwxr-xr-x. 1 root root 5.6M Jul 15 14:06 nvcc
-rwxr-xr-x. 1 root root 268K Jul 15 14:06 fatbinary
-rwxr-xr-x. 1 root root 5.3M Jul 15 14:06 cudafe++
-rwxr-xr-x. 1 root root  79K Jul 15 14:06 bin2c
-rwxr-xr-x. 1 root root 5.5M Jul 15 14:08 nvprof
-rwxr-xr-x. 1 root root  800 Jul 15 14:09 cuda-install-samples-11.4.sh
-rwxr-xr-x. 1 root root  115 Jul 27 15:03 compute-sanitizer
lrwxrwxrwx. 1 root root    3 Jul 28 15:43 nv-nsight-cu-cli -> ncu
lrwxrwxrwx. 1 root root    6 Jul 28 15:43 nv-nsight-cu -> ncu-ui
-rwxr-xr-x. 3 root root 2.6K Jul 28 15:43 ncu-ui
-rwxr-xr-x. 3 root root 3.0K Jul 28 15:43 ncu
-rwxr-xr-x. 1 root root  739 Jul 28 15:43 nsys-ui
-rwxr-xr-x. 3 root root  104 Jul 28 15:43 nsys-exporter
-rwxr-xr-x. 1 root root  746 Jul 28 15:43 nsys
-rwxr-xr-x. 2 root root   82 Jul 28 15:43 nsight-sys
drwxr-xr-x. 2 root root   43 Aug 20 07:56 crt
```

`nvcc` is the NVIDIA CUDA Compiler. Note that `nvcc` is built on `llvm` as [described here](https://developer.nvidia.com/cuda-llvm-compiler). To learn more about an executable, use the help option. For instance: `nvcc --help`.


Let's look at the libraries:

```
$ ls -lL /usr/local/cuda-11.4/lib64/lib*.so

-rwxr-xr-x. 1 root root   2111864 Jul 15 14:08 /usr/local/cuda-11.4/lib64/libaccinj64.so
-rwxr-xr-x. 1 root root 287898200 Jul 15 00:09 /usr/local/cuda-11.4/lib64/libcublasLt.so
-rwxr-xr-x. 1 root root 148055912 Jul 15 00:09 /usr/local/cuda-11.4/lib64/libcublas.so
-rwxr-xr-x. 1 root root    658784 Jul 27 02:08 /usr/local/cuda-11.4/lib64/libcudart.so
-rwxr-xr-x. 1 root root 361300440 Jul 14 23:41 /usr/local/cuda-11.4/lib64/libcufft.so
-rwxr-xr-x. 1 root root    737016 Jul 14 23:41 /usr/local/cuda-11.4/lib64/libcufftw.so
-rwxr-xr-x. 1 root root     34864 Jul 13 18:32 /usr/local/cuda-11.4/lib64/libcufile_rdma.so
-rwxr-xr-x. 1 root root   1263736 Jul 13 18:32 /usr/local/cuda-11.4/lib64/libcufile.so
-rwxr-xr-x. 1 root root   2528064 Jul 15 14:08 /usr/local/cuda-11.4/lib64/libcuinj64.so
-rwxr-xr-x. 1 root root   7306096 Jul 14 23:36 /usr/local/cuda-11.4/lib64/libcupti.so
-rwxr-xr-x. 1 root root  83324336 Jul 15 14:04 /usr/local/cuda-11.4/lib64/libcurand.so
-rwxr-xr-x. 1 root root 239108896 Jul 15 14:17 /usr/local/cuda-11.4/lib64/libcusolverMg.so
-rwxr-xr-x. 1 root root 218905088 Jul 15 14:17 /usr/local/cuda-11.4/lib64/libcusolver.so
-rwxr-xr-x. 1 root root 236863176 Jul 15 14:12 /usr/local/cuda-11.4/lib64/libcusparse.so
-rwxr-xr-x. 1 root root   1573360 Jul 15 14:20 /usr/local/cuda-11.4/lib64/libnppc.so
-rwxr-xr-x. 1 root root  13034288 Jul 15 14:20 /usr/local/cuda-11.4/lib64/libnppial.so
-rwxr-xr-x. 1 root root   6320944 Jul 15 14:20 /usr/local/cuda-11.4/lib64/libnppicc.so
-rwxr-xr-x. 1 root root   9655448 Jul 15 14:20 /usr/local/cuda-11.4/lib64/libnppidei.so
-rwxr-xr-x. 1 root root  67729336 Jul 15 14:20 /usr/local/cuda-11.4/lib64/libnppif.so
-rwxr-xr-x. 1 root root  32134032 Jul 15 14:20 /usr/local/cuda-11.4/lib64/libnppig.so
-rwxr-xr-x. 1 root root   8270664 Jul 15 14:20 /usr/local/cuda-11.4/lib64/libnppim.so
-rwxr-xr-x. 1 root root  31708240 Jul 15 14:20 /usr/local/cuda-11.4/lib64/libnppist.so
-rwxr-xr-x. 1 root root    650664 Jul 15 14:20 /usr/local/cuda-11.4/lib64/libnppisu.so
-rwxr-xr-x. 1 root root   4436592 Jul 15 14:20 /usr/local/cuda-11.4/lib64/libnppitc.so
-rwxr-xr-x. 1 root root  17224952 Jul 15 14:20 /usr/local/cuda-11.4/lib64/libnpps.so
-rwxr-xr-x. 1 root root    700232 Jul 15 00:09 /usr/local/cuda-11.4/lib64/libnvblas.so
-rwxr-xr-x. 1 root root   5149552 Jul 15 14:03 /usr/local/cuda-11.4/lib64/libnvjpeg.so
-rwxr-xr-x. 1 root root  18559928 Jul 14 23:36 /usr/local/cuda-11.4/lib64/libnvperf_host.so
-rwxr-xr-x. 1 root root   4246328 Jul 14 23:36 /usr/local/cuda-11.4/lib64/libnvperf_target.so
-rwxr-xr-x. 1 root root   6883208 Jul 15 14:10 /usr/local/cuda-11.4/lib64/libnvrtc-builtins.so
-rwxr-xr-x. 1 root root  44694608 Jul 15 14:10 /usr/local/cuda-11.4/lib64/libnvrtc.so
-rwxr-xr-x. 1 root root     40136 Jul 15 13:58 /usr/local/cuda-11.4/lib64/libnvToolsExt.so
-rwxr-xr-x. 1 root root     30856 Jul 27 02:08 /usr/local/cuda-11.4/lib64/libOpenCL.so
-rwxr-xr-x. 3 root root    925016 May 29 18:46 /usr/local/cuda-11.4/lib64/libpcsamplingutil.so
```

## cuDNN

There is also the [CUDA Deep Neural Net](https://developer.nvidia.com/cudnn) (cuDNN) library. It is external to the NVIDIA CUDA Toolkit and is used with TensorFlow, for instance, to provide GPU routines for training neural nets.

## Conda Installations

When you install [CuPy](https://cupy.dev), for instance, Conda will include a CUDA Toolkit package (not the development version) and cuDNN:

<pre>
$ module load anaconda3/2020.11
$ conda create --name py-gpu cupy --channel conda-forge

  _libgcc_mutex      conda-forge/linux-64::_libgcc_mutex-0.1-conda_forge
  _openmp_mutex      conda-forge/linux-64::_openmp_mutex-4.5-1_gnu
  ca-certificates    conda-forge/linux-64::ca-certificates-2021.10.8-ha878542_0
  cudatoolkit        conda-forge/linux-64::cudatoolkit-11.4.2-h00f7ccd_9
  cupy               conda-forge/linux-64::cupy-9.5.0-py39h499daff_0
  fastrlock          conda-forge/linux-64::fastrlock-0.6-py39he80948d_1
  ld_impl_linux-64   conda-forge/linux-64::ld_impl_linux-64-2.36.1-hea4e1c9_2
  libblas            conda-forge/linux-64::libblas-3.9.0-12_linux64_openblas
  libcblas           conda-forge/linux-64::libcblas-3.9.0-12_linux64_openblas
  libffi             conda-forge/linux-64::libffi-3.4.2-h9c3ff4c_4
  libgcc-ng          conda-forge/linux-64::libgcc-ng-11.2.0-h1d223b6_11
  libgfortran-ng     conda-forge/linux-64::libgfortran-ng-11.2.0-h69a702a_11
  libgfortran5       conda-forge/linux-64::libgfortran5-11.2.0-h5c6108e_11
  libgomp            conda-forge/linux-64::libgomp-11.2.0-h1d223b6_11
  liblapack          conda-forge/linux-64::liblapack-3.9.0-12_linux64_openblas
  libopenblas        conda-forge/linux-64::libopenblas-0.3.18-pthreads_h8fe5266_0
  libstdcxx-ng       conda-forge/linux-64::libstdcxx-ng-11.2.0-he4da1e4_11
  libzlib            conda-forge/linux-64::libzlib-1.2.11-h36c2ea0_1013
  ncurses            conda-forge/linux-64::ncurses-6.2-h58526e2_4
  numpy              conda-forge/linux-64::numpy-1.21.2-py39hdbf815f_0
  openssl            conda-forge/linux-64::openssl-3.0.0-h7f98852_1
  pip                conda-forge/noarch::pip-21.3-pyhd8ed1ab_0
  python             conda-forge/linux-64::python-3.9.7-hf930737_3_cpython
  python_abi         conda-forge/linux-64::python_abi-3.9-2_cp39
  readline           conda-forge/linux-64::readline-8.1-h46c0cb4_0
  setuptools         conda-forge/linux-64::setuptools-58.2.0-py39hf3d152e_0
  sqlite             conda-forge/linux-64::sqlite-3.36.0-h9cd32fc_2
  tk                 conda-forge/linux-64::tk-8.6.11-h27826a3_1
  tzdata             conda-forge/noarch::tzdata-2021d-he74cb21_0
  wheel              conda-forge/noarch::wheel-0.37.0-pyhd8ed1ab_1
  xz                 conda-forge/linux-64::xz-5.2.5-h516909a_1
  zlib               conda-forge/linux-64::zlib-1.2.11-h36c2ea0_1013
</pre>
