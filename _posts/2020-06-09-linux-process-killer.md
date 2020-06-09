---
title: Fear is the mind-killer, Linux is the process-killer.
layout: post
categories: python linux ml
---

In this post, I'll show how the Linux process killer throws a wrench into multiprocessing with Python, especially in the context of high-memory tasks such as machine learning and data processing.
I'll overview the most straightforward way to do multi-core computation with Python, then show how easily this system can fail, and discuss the reasons for this weakness.

## Python for multiprocessing.
I love Python: it's an excellent foundation for many kinds of projects. 
The language is simple, the libraries are wonderful, and projects are easy to debug and iterate on.

Of course, nothing is perfect -- like any language, there are major issues and "gotchas" that need to be dealt with.
Perhaps the biggest, ugliest problem is the inability to do CPU work in multiple threads, a never-going-away legacy issue imposed by the global interpreter lock (GIL) and the reference counting garbage collecting[^gil] used in standard cPython.

It's important to understand the difference between a process and a thread here, so I'll sketch them out quickly.
A process is an independent program in execution, managed by the operating system.
For example, when you run `python` on the command line, you start a new process.
Each process is independent, and they can not interact with each-other except under strict conditions facilitated by the operating system[^procs].
A thread is sequence of execution *within* a single process.
The threads of a process share access to the memory owned by that process, and can communicate seamlessly with each other.
Threads are supported by OS level primitives, but can be seamlessly manipulated by the process that created them itself[^go-example].

In Python, because of the GIL, no two threads can actually be doing work at the same time.
Multiple threads can be *waiting* at the same time (e.g. for an http request or database call to finish), but not *working* (e.g. doing math).

So how can we do concurrent CPU bound tasks easily in Python?
Quite easily: just don't use threads, use processes.
The `multiprocessing` module in the standard library has excellent tools for managing and composing groups of independent Python processes, allowing us to easily make use of multi core CPUs for parallel computation.

One great class from this module is the `Pool`, which creates an arbitrary (bounded) number of independent Python worker processes and provides a simple interface for submitting jobs to the small army of workers.
For example, imagine replacing `f` with whatever function is doing the heavy lifting of your project in the below:

{% highlight python %}
from multiprocessing import Pool

list_of_int_pairs = [(1,2), (1,3)]

def f(a, b):
    return a + b

p = Pool(8)
async_results = []
for a, b in list_of_int_pairs:
    ar = p.apply_async(f, (a, b))
    async_results.append(ar)
p.close()
p.join()
final_results = [ar.get() for ar in async_results]    
{% endhighlight %}

As you can see, it's extremely easy to spray an arbitrary function out onto multiple cores, and really get your money's worth from that recent CPU upgrade.
There's even a nice `with` syntax for the Pool, which I haven't used here for the sake of being explicit.

I use the `Pool` all the time when I have to process large amounts of data. 
For example, when I need to apply a certain transformation to each line of a 20 GB CSV file, I can very easily cut the job time to a fraction by splitting the job across multiple cores[^future-cores].

## How it all goes wrong.

So how does this all go sideways? Well, it would be bad if, for example, the OS (Linux in this case) decided to suddenly terminate one of the worker processes without informing any of the other processes. 
That might cause the process with the `Pool` to wait forever for an update from one of its (now dead) worker processes.

And that's exactly what will happen if you run out of memory while multiprocessing.

Here's a snippet of code sufficient to cause this problem on a Linux system.
I don't recommend running this, as you should except your system to quickly and repeatedly run out of memory.
This could either live lock your system, or, in the better case, iteratively fill up memory, fail in some way, and repeat eight times.
On my laptop it does the former, and on my desktop it does the latter.

{% highlight python %}
from multiprocessing import Pool
import time
import random

def f():
    l = []
    while True:
        l.append(bytearray(64000000))
        time.sleep(random.random())

p = Pool(8)
for i in range(8):
    p.apply_async(f)
p.close()
p.join()
{% endhighlight %}

As you can see, we're simply starting 8 Python processes, each one doomed to request memory repeatedly until the system runs out of memory.
The short 0 to 1 second sleep between requests is simply to give more of a chance for some safety guards to kick in and potentially avert a system crash (more on that soon).
On my desktop, I saw the following memory usage graph.

![Increasing memory usage graph](/assets/images/process_killer/memory_graph.png "Increasing memory usage graph")

So, assuming the system doesn't crash, what happens to the process that started all of these memory-hungry workers? 
In my experience: nothing happens!
It sits and waits for the workers to signal they have finished their jobs, workers which have all failed.
This is obviously highly undesirable: otherwise working code can fail silently because a memory safeguard was tripped, and your calling code will hang indefinitely.

Why is this happening?

## Linux, the process-killer.

To put it simply, when memory usage starts to get really out of hand, Linux will attempt to kill high-memory using processes in order to avert a complete system crash. 

When configured typically, Linux will allow each process to request as much memory as it asks for, even promising access to memory which does not literally exist on the underlying hardware.
Typically this is fine, but Linux is so generous that it may even promise memory which it needs to manage the system itself, potentially leading to total system crashes (as we saw above).
The OS has to live in memory with everything else, after all.
Under "desperately low memory conditions"[^lwn], the OS will lash out and kill a memory-fat process, preserving memory for the OS itself to run.
The system for doing this, the "oom reaper", decides which process to kill based on a set of heuristics, thus possibly killing a totally innocent process (whatever that means).

What defense does a process have against this reaper?
Each process can be assigned a priority, referred to as the `oom_adj`, which can influence its final "badness score", making it more or less immune to the reaper.
A process with an `oom_adj` of `-17` will not be considered for reaping.
However, one should not necessarily set this for all their memory-hogging processes, as it virtually guarantees that a system crash will occur when these processes grow too large.
Finer grained control can be achieved by specifying groups of processes.

So, why does it work this way? Why doesn't the OS save enough memory for itself to operate, avoiding this unhappy situation?
Like everything, it's a tradeoff.
If you decided to reserve some memory for system usage, the next question would be "how much", which does not have an obvious general answer. 
Memory would be wasted in the case that you chose to reserve too much, and the problem wouldn't be solved in the case where you reserved too little.
The decision most major distributions make is to assume that actually completely filling the system memory is a rare event, and a set or heuristics for killing processes in this rare event is preferable to reducing the overall memory available to programs.

If indeed you fear the reaper, you can fiddle with the kernel variables `vm.overcommit_memory` (and the related `vm.overcommit_ratio` or `vm.overcommit_kbytes`) in order to reserve a set of memory for OS use. 
If this is done correctly, I assume that whenever a processes which request memory when memory is already full will immediately crash (i.e. the call to `malloc` will fail), rather than the OS choosing a process to kill for you when memory is almost full.
This could result in a much worse situation, as the most memory intensive processes may not be cleared out quickly, causing many smaller longer running processes (the "innocents") to fail before the system returns to a stable state.
Seen in this light, isn't it preferable that the OS specifically target the most egregious users of memory for killing, rather than the kinder, gentler, database servers and daemons of the world?

Still, if you choose to go this way, it should help to avoid a live-lock situation, which sometimes occurs when the reaper is not quick enough to kill a greedy program.

One final caveat: if you do try to reserve memory for the system, you will be (implicitly) disabling an extremely useful optimization: the ability for the OS to optimistically over promise on memory.
Normally, requests for memory can be optimistically granted ("sure, you've got your 86 GB") under the assumption that programs will not *typically* use all the memory they request all at once.
The OS keeps track of how much memory has been requested (and how much is truly being used), and invokes the oom killer when the actual usage rises above a danger threshold.
This optimization *must* be disabled if you wish to reserve memory for the OS.
This is because the only way to ensure that limits put in place are actually respected is for the OS to become a strict literalist when it comes to fulfilling process memory requests.
If a process requests 100 MB, the OS will have to consider that 100 MB completely used from the moment it is requested, whether the process ever uses that memory or not.
Therefore, if disabled, processes will almost certainly use more memory on average, and you may find that the overall rate of failures due to running out of memory shoot up.

With all this information, it seems pretty clear to me that in fact the tradeoff is relatively good.
A well tuned set of heuristics will try to kill the processes which are most aggressively demanding memory in order to increase the odds of the longer running more conservative processes can continue about their business.
Given the constraints we've discussed here, I'm struggling to think of a better option.

One final note: if, like me, you've ever left a multi-processing Python program running over night, only to find in the morning the process was simply hanging, you can check `/var/log/kern.log` to see if you've run in to this unfortunate situation.
A quick search for "oom_reaper" should show you which processes were victimized, and when.

## Returning to Python

So far, we've learned that Python must resort to multi-processing to benefit from multi-core computation, and that these independent processes are liable to be unceremoniously killed by the underlying environment, leading to a case where some parts of the program have been killed while others are still active, potentially waiting for them.

What can be done about this?

### A process pool that knows when a worker has been killed.

What if we had a worker process pool that knew when one of its workers had been killed? 
Well, it exists: the `concurrent.futures.ProcessPoolExecutor`.
The API for this class is slightly different than the `multiprocessing.Pool`, but similar enough that you can quickly switch over to it.
To resurrect our example from above:

{% highlight python %}
from concurrent.futures import ProcessPoolExecutor

list_of_int_pairs = [(1,2), (1,3)]

def f(a, b):
    return a + b

p = ProcessPoolExecutor(8)
futures = []
for a, b in list_of_int_pairs:
    future = p.submit(f, a, b)
    futures.append(future)
p.shutdown()
final_results = [f.result() for f in futures]    
{% endhighlight %}

The `ProcessPoolExecutor` will raise a `BrokenProcessPool` exception immediately if one of the worker processes is killed, including in the case where the oom reaper strikes.
Frankly, looking over the documentation, I find it surprising that the documentation for the `multiprocessing` module does not reference `concurrent.futures`  anywhere!
It seems like it would be useful to point readers towards a more modern worker pool implementation, with a simpler API and better guarantees.
I would recommend to use the `ProcessPoolExecutor` if you're approaching a problem with a worker pool.

Switching to `ProcessPoolExecutor` will at least fix the issue of the program hanging on worker death, but it won't stop you from running out of memory under full load if this was already the case.

### Reduce memory usage with streaming

One solution is to try to reduce how much memory each process is using.
There are many techniques for this, but perhaps the simplest one is to design each process as operating on streams of data, where input is read, processed, and written out one line at a time.
Many problems are decomposable in this way, and it's very applicable to the case of CSV processing, for instance.
Storage space is cheap and fast now, so reading in and writing out data to files multiple times in a pipeline is often not a significant performance barrier when memory constrained.

In a future blog post I will likely share how I typically approach this.

### Try Java

Another, perhaps less popular, idea is to switch to a language with better threading support. 
Is this giving up? 
I don't think so -- use the right tool for the job!
In a language like Java or C#, where threads can compute in parallel and share memory, you'll likely be able to save a lot of memory by simply sharing information between threads.
Further, because threads are managed by the runtime in Java, it's easy to write programs in these languages that either "work" or "don't work" as one, instead of the situation in Python where the program can be half alive, half dead.

# Conclusion

In this post I overviewed how simple it is to multiprocess in Python. I then discussed one of the main issues you may run in to when attempting this: the oom reaper.
I then discussed why the reaper strikes, and outlined why (while seemingly arbitrary), this system is quite a nice tradeoff given the constraints on the system.
Finally, I outlined some possible directions for avoiding hanging execution and memory overflows when working with high memory tasks like data processing and machine learning.

I hope it's been interesting and useful!

## Footnotes

[^gil]: See [RealPython](https://realpython.com/python-gil/) for a nice high level overview of this issue. 
[^procs]: This is a necessary safety feature to ensure, for example, your web browser can not influence the execution of your package manager.
[^future-cores]: I'm planning to discuss approaches to this in an upcoming blog post.
[^go-example]: For example, the go runtime creates a number of OS threads, then manages the assigning of work to those threads via an internal thread pool, enabling their co-routine based asynchronous model.
[^lwn]: https://lwn.net/Articles/317814/