# NVIDIA CUDA Toolkit

![NVIDIA CUDA](https://upload.wikimedia.org/wikipedia/en/b/b9/Nvidia_CUDA_Logo.jpg)

The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) provides a comprehensive set of libraries and tools for developing and running GPU-accelerated applications.

List the available modules that are related to CUDA:

```
$ module avail cudatoolkit
------------ /usr/local/share/Modules/modulefiles -------------
cudatoolkit/10.2  cudatoolkit/11.7  cudatoolkit/12.4  
cudatoolkit/11.1  cudatoolkit/12.0  cudatoolkit/12.5  
cudatoolkit/11.3  cudatoolkit/12.2  cudatoolkit/12.6  
cudatoolkit/11.4  cudatoolkit/12.3  
```

Run the following command to see which environment variables the `cudatoolkit` module is modifying:

```
$ $ module show cudatoolkit/12.5
-------------------------------------------------------------------
/usr/local/share/Modules/modulefiles/cudatoolkit/12.5:

module-whatis   {Sets up cudatoolkit125 12.5 in your environment}
prepend-path    PATH /usr/local/cuda-12.5/bin
prepend-path    LD_LIBRARY_PATH /usr/local/cuda-12.5/lib64
prepend-path    LIBRARY_PATH /usr/local/cuda-12.5/lib64
prepend-path    MANPATH /usr/local/cuda-12.5/doc/man
append-path     -d { } LDFLAGS -L/usr/local/cuda-12.5/lib64
append-path     -d { } INCLUDE -I/usr/local/cuda-12.5/include
append-path     CPATH /usr/local/cuda-12.5/include
append-path     -d { } FFLAGS -I/usr/local/cuda-12.5/include
append-path     -d { } LOCAL_LDFLAGS -L/usr/local/cuda-12.5/lib64
append-path     -d { } LOCAL_INCLUDE -I/usr/local/cuda-12.5/include
append-path     -d { } LOCAL_CFLAGS -I/usr/local/cuda-12.5/include
append-path     -d { } LOCAL_FFLAGS -I/usr/local/cuda-12.5/include
append-path     -d { } LOCAL_CXXFLAGS -I/usr/local/cuda-12.5/include
setenv          CUDA_HOME /usr/local/cuda-12.5
-------------------------------------------------------------------
```

Let's look at the files in `/usr/local/cuda-12.5/bin`:

```
$ ls -ltrh /usr/local/cuda-12.5/bin
total 243M
-rwxr-xr-x.  1 root root  49M Apr 15 22:46 nvdisasm
-rwxr-xr-x.  1 root root 688K Apr 15 22:47 cuobjdump
-rwxr-xr-x.  6 root root  11K May 17 18:50 __nvcc_device_query
-rwxr-xr-x. 14 root root  285 May 17 18:50 nvvp
-rwxr-xr-x.  1 root root 111K Jun  6 06:03 nvprune
-rwxr-xr-x.  1 root root  75K Jun  6 06:09 cu++filt
-rwxr-xr-x.  1 root root  30M Jun  6 06:12 ptxas
-rwxr-xr-x.  1 root root  30M Jun  6 06:12 nvlink
-rw-r--r--.  1 root root  465 Jun  6 06:12 nvcc.profile
-rwxr-xr-x.  1 root root  22M Jun  6 06:12 nvcc
-rwxr-xr-x.  1 root root 1.2M Jun  6 06:12 fatbinary
-rwxr-xr-x.  1 root root 7.1M Jun  6 06:12 cudafe++
-rwxr-xr-x.  1 root root  87K Jun  6 06:12 bin2c
-rwxr-xr-x.  1 root root 803K Jun  6 07:25 cuda-gdbserver
-rwxr-xr-x.  1 root root  17M Jun  6 07:25 cuda-gdb-python3.9-tui
-rwxr-xr-x.  1 root root  17M Jun  6 07:25 cuda-gdb-python3.8-tui
-rwxr-xr-x.  1 root root  17M Jun  6 07:25 cuda-gdb-python3.12-tui
-rwxr-xr-x.  1 root root  17M Jun  6 07:25 cuda-gdb-python3.11-tui
-rwxr-xr-x.  1 root root  17M Jun  6 07:25 cuda-gdb-python3.10-tui
-rwxr-xr-x.  1 root root  15M Jun  6 07:25 cuda-gdb-minimal
-rwxr-xr-x.  1 root root 1.9K Jun  6 07:25 cuda-gdb
-rwxr-xr-x.  1 root root 5.8M Jun  6 07:56 nvprof
lrwxrwxrwx.  1 root root    4 Jun  6 08:04 computeprof -> nvvp
-rwxr-xr-x. 11 root root 1.6K Jun 14 19:56 nsight_ee_plugins_manage.sh
-rwxr-xr-x.  1 root root  833 Jun 25 17:54 nsys-ui
-rwxr-xr-x.  1 root root  743 Jun 25 17:54 nsys
-rwxr-xr-x.  5 root root  112 Jul 12 02:21 compute-sanitizer
-rwxr-xr-x.  5 root root 3.6K Jul 26 18:06 ncu-ui
-rwxr-xr-x.  5 root root 3.8K Jul 26 18:06 ncu
-rwxr-xr-x.  4 root root  197 Jul 26 18:06 nsight-sys
drwxr-xr-x.  2 root root   43 Aug 28 10:24 crt
```

`nvcc` is the NVIDIA CUDA Compiler. Note that `nvcc` is built on `llvm` as [described here](https://developer.nvidia.com/cuda-llvm-compiler). To learn more about an executable, use the help option. For instance: `nvcc --help`.


Let's look at the libraries:

```
$ ls -lL /usr/local/cuda-12.5/lib64/lib*.so
-rwxr-xr-x.  1 root root   2412216 Jun  6 07:56 /usr/local/cuda-12.5/lib64/libaccinj64.so
-rwxr-xr-x.  1 root root   1505608 Jun  6 07:30 /usr/local/cuda-12.5/lib64/libcheckpoint.so
-rwxr-xr-x.  1 root root 446820528 Jun  6 06:10 /usr/local/cuda-12.5/lib64/libcublasLt.so
-rwxr-xr-x.  1 root root 104128480 Jun  6 06:10 /usr/local/cuda-12.5/lib64/libcublas.so
-rwxr-xr-x.  1 root root    712032 Jun  6 06:07 /usr/local/cuda-12.5/lib64/libcudart.so
-rwxr-xr-x.  1 root root 276080616 Jun  6 06:16 /usr/local/cuda-12.5/lib64/libcufft.so
-rwxr-xr-x.  1 root root    974920 Jun  6 06:16 /usr/local/cuda-12.5/lib64/libcufftw.so
-rwxr-xr-x.  6 root root     43320 Jun  5 13:57 /usr/local/cuda-12.5/lib64/libcufile_rdma.so
-rwxr-xr-x.  1 root root   2993816 Jun  6 06:53 /usr/local/cuda-12.5/lib64/libcufile.so
-rwxr-xr-x.  1 root root   2832640 Jun  6 07:56 /usr/local/cuda-12.5/lib64/libcuinj64.so
-rwxr-xr-x.  1 root root   7807144 Jun  6 07:30 /usr/local/cuda-12.5/lib64/libcupti.so
-rwxr-xr-x.  1 root root  96529840 Jun  6 06:14 /usr/local/cuda-12.5/lib64/libcurand.so
-rwxr-xr-x.  1 root root  82234792 Jun  6 06:55 /usr/local/cuda-12.5/lib64/libcusolverMg.so
-rwxr-xr-x.  1 root root 122162688 Jun  6 06:55 /usr/local/cuda-12.5/lib64/libcusolver.so
-rwxr-xr-x.  1 root root 294682616 Jun  6 06:29 /usr/local/cuda-12.5/lib64/libcusparse.so
-rwxr-xr-x.  1 root root   1651184 Jun  6 06:37 /usr/local/cuda-12.5/lib64/libnppc.so
-rwxr-xr-x.  1 root root  17736496 Jun  6 06:37 /usr/local/cuda-12.5/lib64/libnppial.so
-rwxr-xr-x.  1 root root   7689032 Jun  6 06:37 /usr/local/cuda-12.5/lib64/libnppicc.so
-rwxr-xr-x.  1 root root  11248792 Jun  6 06:37 /usr/local/cuda-12.5/lib64/libnppidei.so
-rwxr-xr-x.  1 root root 101120104 Jun  6 06:37 /usr/local/cuda-12.5/lib64/libnppif.so
-rwxr-xr-x.  1 root root  41165712 Jun  6 06:37 /usr/local/cuda-12.5/lib64/libnppig.so
-rwxr-xr-x.  1 root root  10703688 Jun  6 06:37 /usr/local/cuda-12.5/lib64/libnppim.so
-rwxr-xr-x.  1 root root  37897296 Jun  6 06:37 /usr/local/cuda-12.5/lib64/libnppist.so
-rwxr-xr-x.  1 root root    724392 Jun  6 06:37 /usr/local/cuda-12.5/lib64/libnppisu.so
-rwxr-xr-x.  1 root root   5595760 Jun  6 06:37 /usr/local/cuda-12.5/lib64/libnppitc.so
-rwxr-xr-x.  1 root root  14169336 Jun  6 06:37 /usr/local/cuda-12.5/lib64/libnpps.so
-rwxr-xr-x.  1 root root    757496 Jun  6 06:10 /usr/local/cuda-12.5/lib64/libnvblas.so
-rwxr-xr-x.  1 root root   2409960 Jun  6 06:08 /usr/local/cuda-12.5/lib64/libnvfatbin.so
-rwxr-xr-x.  1 root root  54560656 Jun  6 06:11 /usr/local/cuda-12.5/lib64/libnvJitLink.so
-rwxr-xr-x.  1 root root   6726448 Jun  6 06:07 /usr/local/cuda-12.5/lib64/libnvjpeg.so
-rwxr-xr-x.  1 root root  28139320 Jun  6 07:30 /usr/local/cuda-12.5/lib64/libnvperf_host.so
-rwxr-xr-x.  1 root root   5579216 Jun  6 07:30 /usr/local/cuda-12.5/lib64/libnvperf_target.so
-rwxr-xr-x.  1 root root   5322632 Jun  6 06:07 /usr/local/cuda-12.5/lib64/libnvrtc-builtins.so
-rwxr-xr-x.  1 root root  61401616 Jun  6 06:07 /usr/local/cuda-12.5/lib64/libnvrtc.so
-rwxr-xr-x. 10 root root     40136 May 17 18:50 /usr/local/cuda-12.5/lib64/libnvToolsExt.so
-rwxr-xr-x. 10 root root     30856 May 17 18:50 /usr/local/cuda-12.5/lib64/libOpenCL.so
-rwxr-xr-x.  1 root root    920920 Jun  6 07:30 /usr/local/cuda-12.5/lib64/libpcsamplingutil.so
```

## cuDNN

There is also the [CUDA Deep Neural Net](https://developer.nvidia.com/cudnn) (cuDNN) library. It is external to the NVIDIA CUDA Toolkit and is used with TensorFlow, for instance, to provide GPU routines for training neural nets. See the available modules with:

```
$ module avail cudnn
```

## Conda Installations

When you install [CuPy](https://cupy.dev), for instance, which is like NumPy for GPUs, Conda will include the CUDA libraries:

<pre>
$ module load anaconda3/2024.6
$ conda create --name cupy-env cupy --channel conda-forge
...
  _libgcc_mutex      conda-forge/linux-64::_libgcc_mutex-0.1-conda_forge 
  _openmp_mutex      conda-forge/linux-64::_openmp_mutex-4.5-2_gnu 
  bzip2              conda-forge/linux-64::bzip2-1.0.8-hd590300_5 
  ca-certificates    conda-forge/linux-64::ca-certificates-2024.7.4-hbcca054_0 
  cuda-nvrtc         conda-forge/linux-64::cuda-nvrtc-12.5.82-he02047a_0 
  cuda-version       conda-forge/noarch::cuda-version-12.5-hd4f0392_3 
  cupy               conda-forge/linux-64::cupy-13.2.0-py312had87585_0 
  cupy-core          conda-forge/linux-64::cupy-core-13.2.0-py312hd074ebb_0 
  fastrlock          conda-forge/linux-64::fastrlock-0.8.2-py312h30efb56_2 
  ld_impl_linux-64   conda-forge/linux-64::ld_impl_linux-64-2.40-hf3520f5_7 
  <b>libblas            conda-forge/linux-64::libblas-3.9.0-22_linux64_openblas 
  <b>libcblas           conda-forge/linux-64::libcblas-3.9.0-22_linux64_openblas 
  <b>libcublas          conda-forge/linux-64::libcublas-12.5.3.2-he02047a_0 
  <b>libcufft           conda-forge/linux-64::libcufft-11.2.3.61-he02047a_0 
  <b>libcurand          conda-forge/linux-64::libcurand-10.3.6.82-he02047a_0 
  <b>libcusolver        conda-forge/linux-64::libcusolver-11.6.3.83-he02047a_0 
  <b>libcusparse        conda-forge/linux-64::libcusparse-12.5.1.3-he02047a_0 </b>
  libexpat           conda-forge/linux-64::libexpat-2.6.2-h59595ed_0 
  libffi             conda-forge/linux-64::libffi-3.4.2-h7f98852_5 
  libgcc-ng          conda-forge/linux-64::libgcc-ng-14.1.0-h77fa898_0 
  libgfortran-ng     conda-forge/linux-64::libgfortran-ng-14.1.0-h69a702a_0 
  libgfortran5       conda-forge/linux-64::libgfortran5-14.1.0-hc5f4f2c_0 
  libgomp            conda-forge/linux-64::libgomp-14.1.0-h77fa898_0 
  liblapack          conda-forge/linux-64::liblapack-3.9.0-22_linux64_openblas 
  libnsl             conda-forge/linux-64::libnsl-2.0.1-hd590300_0 
  libnvjitlink       conda-forge/linux-64::libnvjitlink-12.5.82-he02047a_0 
  libopenblas        conda-forge/linux-64::libopenblas-0.3.27-pthreads_hac2b453_1 
  libsqlite          conda-forge/linux-64::libsqlite-3.46.0-hde9e2c9_0 
  libstdcxx-ng       conda-forge/linux-64::libstdcxx-ng-14.1.0-hc0a3c3a_0 
  libuuid            conda-forge/linux-64::libuuid-2.38.1-h0b41bf4_0 
  libxcrypt          conda-forge/linux-64::libxcrypt-4.4.36-hd590300_1 
  libzlib            conda-forge/linux-64::libzlib-1.3.1-h4ab18f5_1 
  ncurses            conda-forge/linux-64::ncurses-6.5-h59595ed_0 
  numpy              conda-forge/linux-64::numpy-2.0.0-py312h22e1c76_0 
  openssl            conda-forge/linux-64::openssl-3.3.1-h4ab18f5_1 
  pip                conda-forge/noarch::pip-24.0-pyhd8ed1ab_0 
  python             conda-forge/linux-64::python-3.12.4-h194c7f8_0_cpython 
  python_abi         conda-forge/linux-64::python_abi-3.12-4_cp312 
  readline           conda-forge/linux-64::readline-8.2-h8228510_1 
  setuptools         conda-forge/noarch::setuptools-70.1.1-pyhd8ed1ab_0 
  tk                 conda-forge/linux-64::tk-8.6.13-noxft_h4845f30_101 
  tzdata             conda-forge/noarch::tzdata-2024a-h0c530f3_0 
  wheel              conda-forge/noarch::wheel-0.43.0-pyhd8ed1ab_1 
  xz                 conda-forge/linux-64::xz-5.2.6-h166bdaf_0 
</pre>

When using `pip` to do the installation, one needs to load the `cudatoolkit` module since that dependency is assumed to be available on the local system. The Conda approach installs all the dependencies so one does not load the module.
