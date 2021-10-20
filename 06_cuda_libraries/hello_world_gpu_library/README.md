# Building a Simple GPU Library

In this exercise we will construct a "hello world" GPU library called `cumessage` and then link and run a code against it.

### Create the GPU Library

Inspect the files that compose the GPU library:

```bash
$ cd 06_cuda_libraries/hello_world_gpu_library
$ cat cumessage.h
$ cat cumessage.cu
```

`cumessage.h` is the header file. It contains the signature or protocol of one function. That is, the name and the input/output types are specified but the function is not implemented here. The implementation is done in `cumessage.cu`. There is some CUDA code in that file. It will be explained in `07_cuda_kernels`.

Libraries are standalone. That is, there is nothing a present waiting to use our library. We will simply create it and then write a code that can use it. Create the library by running the following commands:

```bash
$ module load cudatoolkit/11.4
$ nvcc -Xcompiler -fPIC -o libcumessage.so -shared cumessage.cu
$ ls -l
```

This will produce `libcumessage.so` which is a GPU library with a single function. Add the option "-v" to the line beginning with `nvcc` to see more details. You will see the `gcc` is being called in addition to `nvcc`.

### Use the GPU Library

and then look at our main routine in `myapp.cu`:

$ cat myapp.cu

Note that `myapp.cu` only needs to know about the inputs and outputs of `GPUfunction`, which are `void` in both cases. Nothing is said about how that function is implemented. That is done entirely in the library.
