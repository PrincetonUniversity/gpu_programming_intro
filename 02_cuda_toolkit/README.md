# NVIDIA CUDA Toolkit

![NVIDIA CUDA](https://upload.wikimedia.org/wikipedia/commons/b/b9/Nvidia_CUDA_Logo.jpg)

The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) provides a comprehensive set of libraries and tools for developing and running GPU-accelerated applications.

List the available modules that are related to CUDA:

```
$ module avail cudatoolkit
-- /usr/local/share/Modules/modulefiles --
cudatoolkit/11.8  cudatoolkit/12.6
cudatoolkit/12.8  cudatoolkit/13.0
```

Run the following command to see which environment variables the `cudatoolkit` module is modifying:

```
$ module show cudatoolkit/12.5
-------------------------------------------------------------------
/usr/local/share/Modules/modulefiles/cudatoolkit/13.0:

module-whatis   {Sets up cudatoolkit130 13.0 in your environment}
prepend-path    PATH /usr/local/cuda-13.0/bin
prepend-path    LD_LIBRARY_PATH /usr/local/cuda-13.0/lib64
prepend-path    LIBRARY_PATH /usr/local/cuda-13.0/lib64
prepend-path    MANPATH /usr/local/cuda-13.0/doc/man
append-path     -d { } LDFLAGS -L/usr/local/cuda-13.0/lib64
append-path     -d { } INCLUDE -I/usr/local/cuda-13.0/include
append-path     CPATH /usr/local/cuda-13.0/include
append-path     -d { } FFLAGS -I/usr/local/cuda-13.0/include
append-path     -d { } LOCAL_LDFLAGS -L/usr/local/cuda-13.0/lib64
append-path     -d { } LOCAL_INCLUDE -I/usr/local/cuda-13.0/include
append-path     -d { } LOCAL_CFLAGS -I/usr/local/cuda-13.0/include
append-path     -d { } LOCAL_FFLAGS -I/usr/local/cuda-13.0/include
append-path     -d { } LOCAL_CXXFLAGS -I/usr/local/cuda-13.0/include
setenv          CUDA_HOME /usr/local/cuda-13.0
-------------------------------------------------------------------
```

Let's look at the files in `/usr/local/cuda-12.5/bin`:

```
$ ls -ltrh /usr/local/cuda-13.0/bin
total 208M
-rwxr-xr-x. 1 root root  36M Jul  7  2015 ptxas
-rwxr-xr-x. 1 root root 115K Jul  7  2015 nvprune
-rwxr-xr-x. 1 root root  37M Jul  7  2015 nvlink
-rwxr-xr-x. 1 root root 4.8M Jul  7  2015 nvdisasm
-rw-r--r--. 1 root root  505 Jul  7  2015 nvcc.profile
-rwxr-xr-x. 1 root root  11K Jul  7  2015 __nvcc_device_query
-rwxr-xr-x. 1 root root  29M Jul  7  2015 nvcc
-rwxr-xr-x. 1 root root 1.6K Jul  7  2015 nsight_ee_plugins_manage.sh
-rwxr-xr-x. 1 root root 1.2M Jul  7  2015 fatbinary
-rwxr-xr-x. 1 root root 685K Jul  7  2015 cuobjdump
-rwxr-xr-x. 1 root root  75K Jul  7  2015 cu++filt
-rwxr-xr-x. 1 root root 750K Jul  7  2015 cuda-gdbserver
-rwxr-xr-x. 1 root root  16M Jul  7  2015 cuda-gdb-python3.9-tui
-rwxr-xr-x. 1 root root  16M Jul  7  2015 cuda-gdb-python3.8-tui
-rwxr-xr-x. 1 root root  16M Jul  7  2015 cuda-gdb-python3.12-tui
-rwxr-xr-x. 1 root root  16M Jul  7  2015 cuda-gdb-python3.11-tui
-rwxr-xr-x. 1 root root  16M Jul  7  2015 cuda-gdb-python3.10-tui
-rwxr-xr-x. 1 root root  15M Jul  7  2015 cuda-gdb-minimal
-rwxr-xr-x. 1 root root 2.1K Jul  7  2015 cuda-gdb
-rwxr-xr-x. 1 root root 8.5M Jul  7  2015 cudafe++
-rwxr-xr-x. 1 root root  112 Jul  7  2015 compute-sanitizer
-rwxr-xr-x. 1 root root  87K Jul  7  2015 bin2c
-rwxr-xr-x. 1 root root  833 Jul 28 02:16 nsys-ui
-rwxr-xr-x. 1 root root  743 Jul 28 02:16 nsys
-rwxr-xr-x. 1 root root  197 Jul 28 02:16 nsight-sys
-rwxr-xr-x. 1 root root 2.7K Jul 28 02:16 ncu-ui
-rwxr-xr-x. 1 root root 2.7K Jul 28 02:16 ncu
drwxr-xr-x. 2 root root   43 Aug 20 06:23 crt
```

`nvcc` is the NVIDIA CUDA Compiler. Note that `nvcc` is built on `llvm` as [described here](https://developer.nvidia.com/cuda-llvm-compiler). To learn more about an executable, use the help option. For instance: `nvcc --help`.


Let's look at the libraries:

```
$ ls -lL /usr/local/cuda-13.0/lib64/lib*.so
-rwxr-xr-x. 1 root root   1382856 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcheckpoint.so
-rwxr-xr-x. 1 root root 538836848 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcublasLt.so
-rwxr-xr-x. 1 root root  52941016 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcublas.so
-rwxr-xr-x. 1 root root    704288 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcudart.so
-rwxr-xr-x. 1 root root 286542352 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcufft.so
-rwxr-xr-x. 1 root root    991704 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcufftw.so
-rwxr-xr-x. 1 root root     43320 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcufile_rdma.so
-rwxr-xr-x. 1 root root   3170800 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcufile.so
-rwxr-xr-x. 1 root root   4156304 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcupti.so
-rwxr-xr-x. 1 root root 132698328 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcurand.so
-rwxr-xr-x. 1 root root 100483432 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcusolverMg.so
-rwxr-xr-x. 1 root root 137006536 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcusolver.so
-rwxr-xr-x. 1 root root 156214040 Jul  7  2015 /usr/local/cuda-13.0/lib64/libcusparse.so
-rwxr-xr-x. 1 root root   1623344 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnppc.so
-rwxr-xr-x. 1 root root  14268016 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnppial.so
-rwxr-xr-x. 1 root root   6416008 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnppicc.so
-rwxr-xr-x. 1 root root   8992728 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnppidei.so
-rwxr-xr-x. 1 root root  66612104 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnppif.so
-rwxr-xr-x. 1 root root  28763856 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnppig.so
-rwxr-xr-x. 1 root root   7812744 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnppim.so
-rwxr-xr-x. 1 root root  27760584 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnppist.so
-rwxr-xr-x. 1 root root   1627392 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnppisu.so
-rwxr-xr-x. 1 root root   4478384 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnppitc.so
-rwxr-xr-x. 1 root root  10995768 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnpps.so
-rwxr-xr-x. 1 root root    733408 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnvblas.so
-rwxr-xr-x. 1 root root   2467432 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnvfatbin.so
-rwxr-xr-x. 1 root root  98802640 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnvJitLink.so
-rwxr-xr-x. 1 root root   5871216 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnvjpeg.so
-rwxr-xr-x. 1 root root  32405040 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnvperf_host.so
-rwxr-xr-x. 1 root root   5304944 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnvperf_target.so
-rwxr-xr-x. 1 root root 109813296 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnvrtc.alt.so
-rwxr-xr-x. 1 root root   4372424 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnvrtc-builtins.alt.so
-rwxr-xr-x. 1 root root   4380616 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnvrtc-builtins.so
-rwxr-xr-x. 1 root root 109329872 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnvrtc.so
-rwxr-xr-x. 1 root root     40160 Jul  7  2015 /usr/local/cuda-13.0/lib64/libnvtx3interop.so
-rwxr-xr-x. 1 root root     30856 Jul  7  2015 /usr/local/cuda-13.0/lib64/libOpenCL.so
-rwxr-xr-x. 1 root root    703904 Jul  7  2015 /usr/local/cuda-13.0/lib64/libpcsamplingutil.so
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
  libcblas           conda-forge/linux-64::libcblas-3.9.0-22_linux64_openblas 
  libcublas          conda-forge/linux-64::libcublas-12.5.3.2-he02047a_0 
  libcufft           conda-forge/linux-64::libcufft-11.2.3.61-he02047a_0 
  libcurand          conda-forge/linux-64::libcurand-10.3.6.82-he02047a_0 
  libcusolver        conda-forge/linux-64::libcusolver-11.6.3.83-he02047a_0 
  libcusparse        conda-forge/linux-64::libcusparse-12.5.1.3-he02047a_0 </b>
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
