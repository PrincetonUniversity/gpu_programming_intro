# A Primer on GPU Programming

## Setup for live workshop

### Point your browser to `https://bit.ly/36g5YUS`

+ Connect to the eduroam wireless network

+ Open a terminal (e.g., Terminal, PowerShell, PuTTY) [<a href="https://researchcomputing.princeton.edu/education/training/hardware-and-software-requirements-picscie-workshops" target="_blank">click here</a> for help]

+ If a faculty member has sponsored an account for you on TigerGPU then please use that for this workshop since we have a limited number of GPUs on the training cluster. Traverse can be used for most of the exercises.

+ Otherwise, please SSH to Adroit in the terminal: `ssh <NetID>@adroit.princeton.edu` [click [here](https://researchcomputing.princeton.edu/faq/why-cant-i-login-to-a-clu) for help]

+ If you are new to Linux then consider using the MyAdroit web portal: [https://myadroit.princeton.edu](https://myadroit.princeton.edu)

+ Clone this repo on your chosen HPC cluster (e.g., Adroit):

   `git clone https://github.com/PrincetonUniversity/gpu_programming_intro`

+ For the live workshop, to get access to the GPU nodes on Adroit, add this line to your Slurm scripts:

   `#SBATCH --reservation=introgpu`
   
+ Because we have a limited number of GPUs on Adroit, keep the total run time limit of your jobs to 30 seconds:

   `#SBATCH --time=00:00:30`

+ To cancel a job use the command `scancel <JobID>` where `<JobID>` can be obtained from the command `squeue -u $USER`.

+ Go to the [main page](https://github.com/PrincetonUniversity/gpu_programming_intro) of this repo
