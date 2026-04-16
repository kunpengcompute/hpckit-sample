#!/bin/bash
# 假定机器共有32个numa节点

# 在32个numa节点上分配大页，该行为需要root权限进行操作
for i in {0..31}
do
echo 2000 >/sys/devices/system/node/node$i/hugepages/hugepages-2048kB/nr_hugepages
done
# 运行时需要绑核，注意确保内存分配的numa是距离绑定的核最近的numa
taskset -c 0 ./memcpy_between_numa_nodes
# 运行结束后将大页清除
for i in {0..31}
do
echo 0 >/sys/devices/system/node/node$i/hugepages/hugepages-2048kB/nr_hugepages
done