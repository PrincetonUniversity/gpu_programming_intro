# NVIDIA CUDA Toolkit

![NVIDIA CUDA](https://upload.wikimedia.org/wikipedia/commons/b/b9/Nvidia_CUDA_Logo.jpg)

The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) provides a comprehensive set of libraries and tools for developing and running GPU-accelerated applications.

List the available modules that are related to CUDA:

```
$ module avail cudatoolkit
-- /usr/local/share/Modules/modulefiles --
cudatoolkit/11.8  cudatoolkit/12.6  cudatoolkit/12.8
cudatoolkit/13.0  cudatoolkit/13.2
```

Run the following command to see which environment variables the `cudatoolkit` module is modifying:

```
$ module show cudatoolkit/13.2
-------------------------------------------------------------------
/usr/local/share/Modules/modulefiles/cudatoolkit/13.2:

module-whatis   {Sets up cudatoolkit132 13.2 in your environment}
prepend-path    PATH /usr/local/cuda-13.2/bin
prepend-path    LD_LIBRARY_PATH /usr/local/cuda-13.2/lib64
prepend-path    LIBRARY_PATH /usr/local/cuda-13.2/lib64
prepend-path    MANPATH /usr/local/cuda-13.2/doc/man
append-path     -d { } LDFLAGS -L/usr/local/cuda-13.2/lib64
append-path     -d { } INCLUDE -I/usr/local/cuda-13.2/include
append-path     CPATH /usr/local/cuda-13.2/include
append-path     -d { } FFLAGS -I/usr/local/cuda-13.2/include
append-path     -d { } LOCAL_LDFLAGS -L/usr/local/cuda-13.2/lib64
append-path     -d { } LOCAL_INCLUDE -I/usr/local/cuda-13.2/include
append-path     -d { } LOCAL_CFLAGS -I/usr/local/cuda-13.2/include
append-path     -d { } LOCAL_FFLAGS -I/usr/local/cuda-13.2/include
append-path     -d { } LOCAL_CXXFLAGS -I/usr/local/cuda-13.2/include
setenv          CUDA_HOME /usr/local/cuda-13.2
-------------------------------------------------------------------
```

Let's look at the files in `/usr/local/cuda-13.2/bin`:

```
$ ls -ltrh /usr/local/cuda-13.2/bin
total 325M
-rwxr-xr-x. 1 root root  91M Jul  7  2015 tileiras
-rwxr-xr-x. 1 root root  45M Jul  7  2015 ptxas
-rwxr-xr-x. 1 root root 1.6M Jul  7  2015 nvprune
-rwxr-xr-x. 1 root root  41M Jul  7  2015 nvlink
-rwxr-xr-x. 1 root root 4.9M Jul  7  2015 nvdisasm
-rw-r--r--. 1 root root  505 Jul  7  2015 nvcc.profile
-rwxr-xr-x. 1 root root  11K Jul  7  2015 __nvcc_device_query
-rwxr-xr-x. 1 root root  33M Jul  7  2015 nvcc
-rwxr-xr-x. 1 root root 1.6K Jul  7  2015 nsight_ee_plugins_manage.sh
-rwxr-xr-x. 1 root root 1.4M Jul  7  2015 fatbinary
-rwxr-xr-x. 1 root root 760K Jul  7  2015 cuobjdump
-rwxr-xr-x. 1 root root  78K Jul  7  2015 cu++filt
-rwxr-xr-x. 1 root root 810K Jul  7  2015 cuda-gdbserver
-rwxr-xr-x. 1 root root  16M Jul  7  2015 cuda-gdb-python3.9-tui
-rwxr-xr-x. 1 root root  16M Jul  7  2015 cuda-gdb-python3.8-tui
-rwxr-xr-x. 1 root root  16M Jul  7  2015 cuda-gdb-python3.12-tui
-rwxr-xr-x. 1 root root  16M Jul  7  2015 cuda-gdb-python3.11-tui
-rwxr-xr-x. 1 root root  16M Jul  7  2015 cuda-gdb-python3.10-tui
-rwxr-xr-x. 1 root root  15M Jul  7  2015 cuda-gdb-minimal
-rwxr-xr-x. 1 root root 2.1K Jul  7  2015 cuda-gdb
-rwxr-xr-x. 1 root root  15M Jul  7  2015 cudafe++
-rwxr-xr-x. 1 root root  112 Jul  7  2015 compute-sanitizer
-rwxr-xr-x. 1 root root  91K Jul  7  2015 bin2c
-rwxr-xr-x. 1 root root  833 Apr  6 23:06 nsys-ui
-rwxr-xr-x. 1 root root  743 Apr  6 23:06 nsys
-rwxr-xr-x. 1 root root  197 Apr  6 23:06 nsight-sys
-rwxr-xr-x. 1 root root 2.7K Apr  6 23:06 ncu-ui
-rwxr-xr-x. 1 root root 2.7K Apr  6 23:06 ncu
drwxr-xr-x. 2 root root   43 Jun  1 10:50 crt
```

`nvcc` is the NVIDIA CUDA Compiler. Note that `nvcc` is built on `llvm` as [described here](https://developer.nvidia.com/cuda-llvm-compiler). To learn more about an executable, use the help option. For instance: `nvcc --help`.


Let's look at the libraries:

```
$ ls -lL /usr/local/cuda-13.2/lib64/lib*.so
-rwxr-xr-x. 1 root root   1646344 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcheckpoint.so
-rwxr-xr-x. 1 root root 508260928 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcublasLt.so
-rwxr-xr-x. 1 root root  54198912 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcublas.so
-rwxr-xr-x. 1 root root    773920 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcudart.so
-rwxr-xr-x. 1 root root 290858264 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcufft.so
-rwxr-xr-x. 1 root root    991704 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcufftw.so
-rwxr-xr-x. 1 root root     48672 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcufile_rdma.so
-rwxr-xr-x. 1 root root   3757888 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcufile.so
-rwxr-xr-x. 1 root root    408824 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcuobjclient.so
-rwxr-xr-x. 1 root root   5912448 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcupti.so
-rwxr-xr-x. 1 root root 133030104 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcurand.so
-rwxr-xr-x. 1 root root 118154968 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcusolverMg.so
-rwxr-xr-x. 1 root root 156451168 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcusolver.so
-rwxr-xr-x. 1 root root 167351288 Jul  7  2015 /usr/local/cuda-13.2/lib64/libcusparse.so
-rwxr-xr-x. 1 root root   1692976 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnppc.so
-rwxr-xr-x. 1 root root  13522544 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnppial.so
-rwxr-xr-x. 1 root root   6973120 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnppicc.so
-rwxr-xr-x. 1 root root   8644568 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnppidei.so
-rwxr-xr-x. 1 root root  60353416 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnppif.so
-rwxr-xr-x. 1 root root  26158800 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnppig.so
-rwxr-xr-x. 1 root root   6841992 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnppim.so
-rwxr-xr-x. 1 root root  27039688 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnppist.so
-rwxr-xr-x. 1 root root   1692928 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnppisu.so
-rwxr-xr-x. 1 root root   4281776 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnppitc.so
-rwxr-xr-x. 1 root root  10324048 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnpps.so
-rwxr-xr-x. 1 root root    737960 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnvblas.so
-rwxr-xr-x. 1 root root   1374592 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnvfatbin.so
-rwxr-xr-x. 1 root root 100441296 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnvJitLink.so
-rwxr-xr-x. 1 root root   6043248 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnvjpeg.so
-rwxr-xr-x. 1 root root  33803312 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnvperf_host.so
-rwxr-xr-x. 1 root root   5990576 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnvperf_target.so
-rwxr-xr-x. 1 root root   4442056 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnvrtc-builtins.so
-rwxr-xr-x. 1 root root 115324240 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnvrtc.so
-rwxr-xr-x. 1 root root     40160 Jul  7  2015 /usr/local/cuda-13.2/lib64/libnvtx3interop.so
-rwxr-xr-x. 1 root root     30856 Jul  7  2015 /usr/local/cuda-13.2/lib64/libOpenCL.so
-rwxr-xr-x. 1 root root    703904 Jul  7  2015 /usr/local/cuda-13.2/lib64/libpcsamplingutil.so
```

## Conda Installations

When you install [CuPy](https://cupy.dev), for instance, which is like NumPy for GPUs, Conda will include the CUDA libraries:

<pre>
$ module load anaconda3/2025.12
$ conda create --name cupy-env cupy --channel conda-forge
...
  _openmp_mutex      conda-forge/linux-64::_openmp_mutex-4.5-20_gnu 
  bzip2              conda-forge/linux-64::bzip2-1.0.8-hda65f42_9 
  ca-certificates    conda-forge/noarch::ca-certificates-2026.6.17-hbd8a1cb_0 
  cuda-cccl_linux-64 conda-forge/noarch::cuda-cccl_linux-64-13.3.3.4.1-ha770c72_0 
  cuda-cudart-dev_l~ conda-forge/noarch::cuda-cudart-dev_linux-64-13.3.29-h376f20c_0 
  cuda-cudart-stati~ conda-forge/noarch::cuda-cudart-static_linux-64-13.3.29-h376f20c_0 
  cuda-cudart_linux~ conda-forge/noarch::cuda-cudart_linux-64-13.3.29-h376f20c_0 
  cuda-nvrtc         conda-forge/linux-64::cuda-nvrtc-13.3.33-hecca717_0 
  cuda-pathfinder    conda-forge/noarch::cuda-pathfinder-1.5.6-pyhc364b38_0 
  cuda-version       conda-forge/noarch::cuda-version-13.3-hcbadf70_3 
  cupy               conda-forge/linux-64::cupy-14.1.1-py314hdea9c46_0 
  cupy-core          conda-forge/linux-64::cupy-core-14.1.1-py314hcd3b49b_0 
  ld_impl_linux-64   conda-forge/linux-64::ld_impl_linux-64-2.45.1-default_hbd61a6d_102 
  <b>libblas            conda-forge/linux-64::libblas-3.11.0-8_h4a7cf45_openblas 
  libcblas           conda-forge/linux-64::libcblas-3.11.0-8_h0358290_openblas 
  libcublas          conda-forge/linux-64::libcublas-13.6.0.2-h676940d_0 
  libcufft           conda-forge/linux-64::libcufft-12.3.0.29-hecca717_0 
  libcurand          conda-forge/linux-64::libcurand-10.4.3.29-h676940d_0 
  libcusolver        conda-forge/linux-64::libcusolver-12.2.6.9-h676940d_0 
  libcusparse        conda-forge/linux-64::libcusparse-12.8.2.51-hecca717_0</b>
  libexpat           conda-forge/linux-64::libexpat-2.8.1-hecca717_1 
  libffi             conda-forge/linux-64::libffi-3.5.2-h3435931_0 
  libgcc             conda-forge/linux-64::libgcc-15.2.0-he0feb66_19 
  libgfortran        conda-forge/linux-64::libgfortran-15.2.0-h69a702a_19 
  libgfortran5       conda-forge/linux-64::libgfortran5-15.2.0-h68bc16d_19 
  libgomp            conda-forge/linux-64::libgomp-15.2.0-he0feb66_19 
  liblapack          conda-forge/linux-64::liblapack-3.11.0-8_h47877c9_openblas 
  liblzma            conda-forge/linux-64::liblzma-5.8.3-hb03c661_0 
  libmpdec           conda-forge/linux-64::libmpdec-4.0.0-hb03c661_1 
  libnvjitlink       conda-forge/linux-64::libnvjitlink-13.3.33-hecca717_0 
  libopenblas        conda-forge/linux-64::libopenblas-0.3.33-pthreads_h94d23a6_0 
  libsqlite          conda-forge/linux-64::libsqlite-3.53.3-h0c1763c_0 
  libstdcxx          conda-forge/linux-64::libstdcxx-15.2.0-h934c35e_19 
  libuuid            conda-forge/linux-64::libuuid-2.42.2-h5347b49_0 
  libzlib            conda-forge/linux-64::libzlib-1.3.2-h25fd6f3_2 
  ncurses            conda-forge/linux-64::ncurses-6.6-hdb14827_0 
  numpy              conda-forge/linux-64::numpy-2.5.1-py314h2b28147_0 
  openssl            conda-forge/linux-64::openssl-3.6.3-h35e630c_0 
  pip                conda-forge/noarch::pip-26.1.2-pyh145f28c_0 
  python             conda-forge/linux-64::python-3.14.6-habeac84_100_cp314 
  python_abi         conda-forge/noarch::python_abi-3.14-8_cp314 
  readline           conda-forge/linux-64::readline-8.3-h853b02a_0 
  tk                 conda-forge/linux-64::tk-8.6.13-noxft_h366c992_103 
  tzdata             conda-forge/noarch::tzdata-2025c-hc9c84f9_1 
  zstd               conda-forge/linux-64::zstd-1.5.7-hb78ec9c_6
</pre>

When using `pip` to do the installation, one needs to load the `cudatoolkit` module since that dependency is assumed to be available on the local system. The Conda approach installs all the dependencies so one does not load the module.
