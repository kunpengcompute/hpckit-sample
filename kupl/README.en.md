# kupl-sample

#### Description

Kupl-sample provides a set of cases using the kupl library .

#### Software Architecture

Each file has sevral sub-files, each sub-file have sample(include demo.cpp, makefile and readme)

example:

    mt_gemm_fp64

    compute_memcpy_overlap

    kernel_parallelism_queue

    dynamic_parallelism_scaling

    local_dependency

    parallel

    zgemm_restrain

memory:

    memcpy

    memcpy_async

    memcpy2d

    memcpy_between_numa_nodes

    shm

    hbw

mma:

    mma_fp64

    mma_bf16

    mma_int8

    copy_int8

mt:

    graph

    graph_lambda

    parallel_for_1d

    parallel_for_ad_lambda

    parallel_for_3d

    queue_event_dependency

    queue_submit

#### Installation

1.  Install the latest HPCKit

    Download：https://www.hikunpeng.com/developer/hpc/hpckit-download
    
    Install：https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/topic_0000001806090516.html

    Using guide：https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_010.html

2.  Download this sample

#### Instructions

Change directory to the target sample

    1. using the script command: `sh build.sh`

    2. Follow the description of tile to build the code and run the demo,
       or use the makefile which descripts in sub-folder readme file

#### Contribution

1.  Fork the repository
2.  Create master branch
3.  Commit your code
4.  Create Pull Request


#### Gitee Feature

1.  You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
