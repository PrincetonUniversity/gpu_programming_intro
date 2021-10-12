# The CUDA Toolkit

The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) provides a comprehensive set of libraries and tools for creating GPU-accelerated applications.

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

Let's look at the files in /usr/local/cuda-10.1/bin:

```
$ ls -ltrh /usr/local/cuda-10.1/bin

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

Note that `nvcc` is built on `llvm` as [described here](https://developer.nvidia.com/cuda-llvm-compiler). To learn more about an executable, use the help option. For instance: `nvcc --help`.


Let's look at the libraries:

```
$ ls -lL /usr/local/cuda-10.1/lib64/lib*.so

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

There is also the CUDA Deep Neural Net library. It is external to the NVIDIA CUDA Toolkit and is used with TensorFlow, for instance, to provide GPU routines for training neural nets.

## Conda Installations

When you install CuPy, for instance, Conda will include a CUDA Toolkit package (not the development version):

<pre>
$ module load anaconda3/2020.11
$ conda create --prefix /scratch/network/$USER/py-gpu cupy

  _libgcc_mutex      pkgs/main/linux-64::_libgcc_mutex-0.1-main
  blas               pkgs/main/linux-64::blas-1.0-mkl
  ca-certificates    pkgs/main/linux-64::ca-certificates-2020.1.1-0
  certifi            pkgs/main/linux-64::certifi-2020.4.5.1-py37_0
  <b>cudatoolkit        pkgs/main/linux-64::cudatoolkit-10.0.130-0</b>
  cudnn              pkgs/main/linux-64::cudnn-7.6.5-cuda10.0_0
  cupy               pkgs/main/linux-64::cupy-6.0.0-py37hc0ce245_0
  fastrlock          pkgs/main/linux-64::fastrlock-0.4-py37he6710b0_0
  intel-openmp       pkgs/main/linux-64::intel-openmp-2020.0-166
  ld_impl_linux-64   pkgs/main/linux-64::ld_impl_linux-64-2.33.1-h53a641e_7
  libedit            pkgs/main/linux-64::libedit-3.1.20181209-hc058e9b_0
  libffi             pkgs/main/linux-64::libffi-3.2.1-hd88cf55_4
  libgcc-ng          pkgs/main/linux-64::libgcc-ng-9.1.0-hdf63c60_0
  libgfortran-ng     pkgs/main/linux-64::libgfortran-ng-7.3.0-hdf63c60_0
  libstdcxx-ng       pkgs/main/linux-64::libstdcxx-ng-9.1.0-hdf63c60_0
  mkl                pkgs/main/linux-64::mkl-2020.0-166
  mkl-service        pkgs/main/linux-64::mkl-service-2.3.0-py37he904b0f_0
  mkl_fft            pkgs/main/linux-64::mkl_fft-1.0.15-py37ha843d7b_0
  mkl_random         pkgs/main/linux-64::mkl_random-1.1.0-py37hd6b4f25_0
  nccl               pkgs/main/linux-64::nccl-1.3.5-cuda10.0_0
  ncurses            pkgs/main/linux-64::ncurses-6.2-he6710b0_0
  numpy              pkgs/main/linux-64::numpy-1.18.1-py37h4f9e942_0
  numpy-base         pkgs/main/linux-64::numpy-base-1.18.1-py37hde5b4d6_1
  openssl            pkgs/main/linux-64::openssl-1.1.1g-h7b6447c_0
  pip                pkgs/main/linux-64::pip-20.0.2-py37_1
  python             pkgs/main/linux-64::python-3.7.7-hcf32534_0_cpython
  readline           pkgs/main/linux-64::readline-8.0-h7b6447c_0
  setuptools         pkgs/main/linux-64::setuptools-46.1.3-py37_0
  six                pkgs/main/linux-64::six-1.14.0-py37_0
  sqlite             pkgs/main/linux-64::sqlite-3.31.1-h62c20be_1
  tk                 pkgs/main/linux-64::tk-8.6.8-hbc83047_0
  wheel              pkgs/main/linux-64::wheel-0.34.2-py37_0
  xz                 pkgs/main/linux-64::xz-5.2.5-h7b6447c_0
  zlib               pkgs/main/linux-64::zlib-1.2.11-h7b6447c_3
</pre>
