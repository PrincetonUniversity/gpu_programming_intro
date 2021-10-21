# Building a Simple GPU Library

In this exercise we will construct a "hello world" GPU library called `cumessage` and then link and run a code against it.

### Create the GPU Library

Inspect the files that compose the GPU library:

```bash
$ cd 06_cuda_libraries/hello_world_gpu_library
$ cat cumessage.h
$ cat cumessage.cu
```

`cumessage.h` is the header file. It contains the signature or protocol of one function. That is, the name and the input/output types are specified but the function body is not implemented here. The implementation is done in `cumessage.cu`. There is some CUDA code in that file. It will be explained in `07_cuda_kernels`.

Libraries are standalone. That is, there is nothing at present waiting to use our library. We will simply create it and then write a code that can use it. Create the library by running the following commands:

```bash
$ module load cudatoolkit/11.4
$ nvcc -Xcompiler -fPIC -o libcumessage.so -shared cumessage.cu
$ ls -ltr
```

This will produce `libcumessage.so` which is a GPU library with a single function. Add the option "-v" to the line beginning with `nvcc` above to see more details. You will see that `gcc` is being called.

### Use the GPU Library

Take a look at our simple code in `myapp.cu` that will use our GPU library:

```bash
$ cat myapp.cu
```

Once again, note that `myapp.cu` only needs to know about the inputs and outputs of `GPUfunction` through the header file. Nothing is known to `myapp.cu` about how that function is implemented.

Compile the main routine against our GPU library:

```
$ nvcc -I. -o myapp myapp.cu -L. -lcudart -lcumessage 
$ ls -ltr
```

This will produce `myapp` which is a GPU application that links against our GPU library `libcumessage.so`:

```
$ env LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ldd myapp
  linux-vdso.so.1 (0x00007fffdaf61000)
  libcumessage.so => ./libcumessage.so (0x000014d68450a000)
  libcudart.so.11.0 => /usr/local/cuda-11.4/lib64/libcudart.so.11.0 (0x000014d684268000)
  librt.so.1 => /lib64/librt.so.1 (0x000014d684060000)
  libpthread.so.0 => /lib64/libpthread.so.0 (0x000014d683e40000)
  libdl.so.2 => /lib64/libdl.so.2 (0x000014d683c3c000)
  libstdc++.so.6 => /lib64/libstdc++.so.6 (0x000014d6838a7000)
  libm.so.6 => /lib64/libm.so.6 (0x000014d683525000)
  libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x000014d68330d000)
  libc.so.6 => /lib64/libc.so.6 (0x000014d682f48000)
  /lib64/ld-linux-x86-64.so.2 (0x000014d6847a9000)
  ```
Finally, submit the job and inspect the output:
  
```
$ sbatch job.slurm
$ cat slurm-*.out
  Hello world from the CPU.
  Hello world from the GPU.
```
