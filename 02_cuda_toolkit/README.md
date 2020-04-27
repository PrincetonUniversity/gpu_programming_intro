# The CUDA Toolkit

The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) provides a development environment for creating high performance GPU-accelerated applications. It provides a comprehensive set of tools and libraries.

List the available modules that are related to CUDA:

```
$ module avail cu

------------------------------------- /usr/local/share/Modules/modulefiles ---------------------------------
cudann/7.0            cudatoolkit/10.1      cudatoolkit/9.2       cudnn/cuda-10.1/7.6.3 cudnn/cuda-9.0/7.1.2
cudann/cuda-7.5/5.1   cudatoolkit/7.5       cudnn/cuda-10.0/7.3.1 cudnn/cuda-8.0/6.0    cudnn/cuda-9.0/7.3.1
cudann/cuda-8.0/5.0   cudatoolkit/8.0       cudnn/cuda-10.0/7.5.0 cudnn/cuda-8.0/7.0    cudnn/cuda-9.1/7.1.2
cudann/cuda-8.0/5.1   cudatoolkit/9.0       cudnn/cuda-10.0/7.6.3 cudnn/cuda-8.0/7.0.3  cudnn/cuda-9.2/7.1.4
cudatoolkit/10.0      cudatoolkit/9.1       cudnn/cuda-10.1/7.5.0 cudnn/cuda-9.0/7.0.3  cudnn/cuda-9.2/7.3.1
```

Run the following command to see which environment variables the `cudatoolkit` module is modifying:

```
$ module show cudatoolkit/10.1
-------------------------------------------------------------------
/usr/local/share/Modules/modulefiles/cudatoolkit/10.1:

module-whatis	 Sets up cudatoolkit101 10.1 in your environment 
prepend-path	 PATH /usr/local/cuda-10.1/bin 
prepend-path	 LD_LIBRARY_PATH /usr/local/cuda-10.1/lib64 
prepend-path	 LIBRARY_PATH /usr/local/cuda-10.1/lib64 
prepend-path	 MANPATH /usr/local/cuda-10.1/doc/man 
append-path	 -d   LDFLAGS -L/usr/local/cuda-10.1/lib64 
append-path	 -d   INCLUDE -I/usr/local/cuda-10.1/include 
append-path	 CPATH /usr/local/cuda-10.1/include 
append-path	 -d   FFLAGS -I/usr/local/cuda-10.1/include 
append-path	 -d   LOCAL_LDFLAGS -L/usr/local/cuda-10.1/lib64 
append-path	 -d   LOCAL_INCLUDE -I/usr/local/cuda-10.1/include 
append-path	 -d   LOCAL_CFLAGS -I/usr/local/cuda-10.1/include 
append-path	 -d   LOCAL_FFLAGS -I/usr/local/cuda-10.1/include 
append-path	 -d   LOCAL_CXXFLAGS -I/usr/local/cuda-10.1/include 
```

Let's look at the files in /usr/local/cuda-10.1/bin:

```
$ ls -ltrh /usr/local/cuda-10.1/bin

-rw-r--r--. 4 root root  393 Apr 12  2018 nvcc.profile
-rwxr-xr-x. 7 root root  215 Apr 12  2018 nvvp
-rwxr-xr-x. 7 root root  219 Apr 12  2018 nsight
-rwxr-xr-x. 1 root root 6.1M Apr 24  2019 ptxas
-rwxr-xr-x. 1 root root  85K Apr 24  2019 nvprune
-rwxr-xr-x. 1 root root 9.7M Apr 24  2019 nvprof
-rwxr-xr-x. 1 root root 6.3M Apr 24  2019 nvlink
-rwxr-xr-x. 1 root root  22M Apr 24  2019 nvdisasm
-rwxr-xr-x. 1 root root 191K Apr 24  2019 nvcc
-rwxr-xr-x. 1 root root 129K Apr 24  2019 fatbinary
-rwxr-xr-x. 1 root root 182K Apr 24  2019 cuobjdump
-rwxr-xr-x. 1 root root 389K Apr 24  2019 cuda-memcheck
-rwxr-xr-x. 1 root root 4.4M Apr 24  2019 cudafe++
-rwxr-xr-x. 1 root root  67K Apr 24  2019 bin2c
-rwxr-xr-x. 1 root root 1.1M Apr 24  2019 gpu-library-advisor
-rwxr-xr-x. 1 root root 569K Apr 24  2019 cuda-gdbserver
-rwxr-xr-x. 1 root root 8.5M Apr 24  2019 cuda-gdb
-rwxr-xr-x. 1 root root 1.6K Apr 24  2019 nsight_ee_plugins_manage.sh
-rwxr-xr-x. 1 root root  800 May  6 19:19 cuda-install-samples-10.1.sh
drwxr-xr-x. 2 root root   43 Jun 26 10:30 crt
lrwxrwxrwx. 1 root root    4 Jun 26 10:32 computeprof -> nvvp
```

Note that `nvcc` is built on `llvm` as [described here](https://developer.nvidia.com/cuda-llvm-compiler). To learn more about an executable, run `module load cudatoolkit` then use the help option, for instance: `nvcc --help`.


Let's look at the libraries:

```
$ ls -lL /usr/local/cuda-10.1/lib64/lib*.so

-rwxr-xr-x. 1 root root   7410800 Apr 24  2019 /usr/local/cuda-10.1/lib64/libaccinj64.so
-rwxr-xr-x. 1 root root    504480 Apr 24  2019 /usr/local/cuda-10.1/lib64/libcudart.so
-rwxr-xr-x. 1 root root 138190088 Apr 24  2019 /usr/local/cuda-10.1/lib64/libcufft.so
-rwxr-xr-x. 1 root root    500568 Apr 24  2019 /usr/local/cuda-10.1/lib64/libcufftw.so
-rwxr-xr-x. 1 root root   7797208 Apr 24  2019 /usr/local/cuda-10.1/lib64/libcuinj64.so
-rwxr-xr-x. 1 root root  59812280 Apr 24  2019 /usr/local/cuda-10.1/lib64/libcurand.so
-rwxr-xr-x. 1 root root 182470736 Apr 24  2019 /usr/local/cuda-10.1/lib64/libcusolver.so
-rwxr-xr-x. 1 root root 122069896 Apr 24  2019 /usr/local/cuda-10.1/lib64/libcusparse.so
-rwxr-xr-x. 1 root root    497176 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnppc.so
-rwxr-xr-x. 1 root root  11856408 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnppial.so
-rwxr-xr-x. 1 root root   4042328 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnppicc.so
-rwxr-xr-x. 1 root root   1344888 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnppicom.so
-rwxr-xr-x. 1 root root   8178496 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnppidei.so
-rwxr-xr-x. 1 root root  47549464 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnppif.so
-rwxr-xr-x. 1 root root  24783768 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnppig.so
-rwxr-xr-x. 1 root root   6281456 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnppim.so
-rwxr-xr-x. 1 root root  18528504 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnppist.so
-rwxr-xr-x. 1 root root    483784 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnppisu.so
-rwxr-xr-x. 1 root root   3049016 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnppitc.so
-rwxr-xr-x. 1 root root   8795936 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnpps.so
-rwxr-xr-x. 1 root root 165605760 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnvgraph.so
-rwxr-xr-x. 1 root root   3376376 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnvjpeg.so
-rwxr-xr-x. 1 root root   4748560 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnvrtc-builtins.so
-rwxr-xr-x. 1 root root  21784872 Apr 24  2019 /usr/local/cuda-10.1/lib64/libnvrtc.so
-rwxr-xr-x. 5 root root     37240 Apr 12  2018 /usr/local/cuda-10.1/lib64/libnvToolsExt.so
-rwxr-xr-x. 2 root root     27096 Apr 24  2019 /usr/local/cuda-10.1/lib64/libOpenCL.so
```

## cuDNN

There is also the CUDA Deep Neural Net library. It is external to the NVIDIA CUDA Toolkit and is used with TensorFlow, for instance, to provide GPU routines for training neural nets.

```
module show cudnn
```

## Conda Installations

When you install CuPy, for instance, Conda will include a CUDA Toolkit package (not the development version):

<pre>
$ module load anaconda3
$ conda create --prefix /scratch/network/$USER/py-gpu cupy

 _libgcc_mutex      pkgs/main/linux-64::_libgcc_mutex-0.1-main
  blas               pkgs/main/linux-64::blas-1.0-mkl
  ca-certificates    pkgs/main/linux-64::ca-certificates-2019.10.16-0
  certifi            pkgs/main/linux-64::certifi-2019.9.11-py37_0
  <b>cudatoolkit        pkgs/main/linux-64::cudatoolkit-10.0.130-0</b>
  cudnn              pkgs/main/linux-64::cudnn-7.6.0-cuda10.0_0
  cupy               pkgs/main/linux-64::cupy-6.0.0-py37hc0ce245_0
  fastrlock          pkgs/main/linux-64::fastrlock-0.4-py37he6710b0_0
  intel-openmp       pkgs/main/linux-64::intel-openmp-2019.4-243
  libedit            pkgs/main/linux-64::libedit-3.1.20181209-hc058e9b_0
  libffi             pkgs/main/linux-64::libffi-3.2.1-hd88cf55_4
  libgcc-ng          pkgs/main/linux-64::libgcc-ng-9.1.0-hdf63c60_0
  libgfortran-ng     pkgs/main/linux-64::libgfortran-ng-7.3.0-hdf63c60_0
  libstdcxx-ng       pkgs/main/linux-64::libstdcxx-ng-9.1.0-hdf63c60_0
  mkl                pkgs/main/linux-64::mkl-2019.4-243
  mkl-service        pkgs/main/linux-64::mkl-service-2.3.0-py37he904b0f_0
  mkl_fft            pkgs/main/linux-64::mkl_fft-1.0.14-py37ha843d7b_0
  mkl_random         pkgs/main/linux-64::mkl_random-1.1.0-py37hd6b4f25_0
  nccl               pkgs/main/linux-64::nccl-1.3.5-cuda10.0_0
  ncurses            pkgs/main/linux-64::ncurses-6.1-he6710b0_1
  numpy              pkgs/main/linux-64::numpy-1.17.2-py37haad9e8e_0
  numpy-base         pkgs/main/linux-64::numpy-base-1.17.2-py37hde5b4d6_0
  openssl            pkgs/main/linux-64::openssl-1.1.1d-h7b6447c_3
  pip                pkgs/main/linux-64::pip-19.3.1-py37_0
  python             pkgs/main/linux-64::python-3.7.5-h0371630_0
  readline           pkgs/main/linux-64::readline-7.0-h7b6447c_5
  setuptools         pkgs/main/linux-64::setuptools-41.4.0-py37_0
  six                pkgs/main/linux-64::six-1.12.0-py37_0
  sqlite             pkgs/main/linux-64::sqlite-3.30.1-h7b6447c_0
  tk                 pkgs/main/linux-64::tk-8.6.8-hbc83047_0
  wheel              pkgs/main/linux-64::wheel-0.33.6-py37_0
  xz                 pkgs/main/linux-64::xz-5.2.4-h14c3975_4
  zlib               pkgs/main/linux-64::zlib-1.2.11-h7b6447c_3
</pre>
