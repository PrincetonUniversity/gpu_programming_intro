# Building a Simple GPU Library

In this exercise we will construct a "hello world" GPU library called `cumessage` and then link and run a code against it.

### Create the Library

Inspect the files that compose the GPU library (`cumessage.h` and `cumessage.cu`) and then look at our main routine in `myapp.cu`:

```bash
$ cd 06_cuda_libraries/hello_world_gpu_library
$ cat cumessage.h
$ cat cumessage.cu
$ cat myapp.cu
```

Note that `myapp.cu` only needs to know about the inputs and outputs of `GPUfunction`, which are `void` in both cases. Nothing is said about how that function is implemented. That is done entirely in the library.

```bash
$ module load cudatoolkit/11.4
$ nvcc -Xcompiler -fPIC -o libcumessage.so -shared cumessage.cu
$ ls -l
```

This will produce `libcumessage.so` which is a GPU library with a single function. Add the option "-v" to the line beginning with `nvcc` to see more details. You will see the `gcc` is being called in addition to `nvcc`.
