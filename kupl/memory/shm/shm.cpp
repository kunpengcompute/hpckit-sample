/*
    KUPL 共享内存管理 Demo
    编译命令: mpicc -lkupl shm.cpp -o shm
    运行命令: mpirun -np 2 -x UCX_TLS=sm shm
*/

#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include "kupl.h"
#include <string.h>

// 创建 kupl 通信域需要的回调函数 1
static int oob_barrier_callback(void *group)
{
    return MPI_Barrier((MPI_Comm)group);
}

// 创建 kupl 通信域需要的回调函数 2
static int oob_allgather_callback(
    const void *sendbuf, void *recvbuf, int size, void *group, kupl_shm_datatype_t datatype)
{
    switch (datatype) {
        case KUPL_SHM_DATATYPE_CHAR:
            return MPI_Allgather(sendbuf, size, MPI_CHAR, recvbuf, size, MPI_CHAR, (MPI_Comm)group);
        default:
            fprintf(stderr, "not support datatype");
            return KUPL_ERROR;
    }
}

int main(int argc, char *argv[])
{
    // 初始化 MPI 环境
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    // 获取 MPI 通信域大小
    int world_size;
    MPI_Comm_size(comm, &world_size);

    // 获取进程 rank 号
    int world_rank;
    MPI_Comm_rank(comm, &world_rank);

    // 获取进程 pid 号
    int pid = getpid();

    // 创建 kupl 通信域
    kupl_shm_oob_cb_t oob_cbs;
    kupl_shm_oob_cb_h oob_cbs_h = &oob_cbs;
    oob_cbs_h->oob_allgather = oob_allgather_callback;
    oob_cbs_h->oob_barrier = oob_barrier_callback;
    kupl_shm_comm_h kupl_comm;
    kupl_shm_comm_create(world_size, world_rank, pid, oob_cbs_h, (void *)comm, &kupl_comm);

    // 申请共享内存
    kupl_shm_win_h win;
    void *local_buffer;
    int count = 4;
    size_t buf_size = count * sizeof(int);
    kupl_shm_win_alloc(buf_size, kupl_comm, &local_buffer, &win);

    // 进程 0 对本进程 buffer 赋值
    if (world_rank == 0) {
        for (int i = 0; i < count; i++) {
            ((int *)local_buffer)[i] = i;
        }
    }
    MPI_Barrier(comm);

    // 所有其他进程获取进程 0 的 buffer，进行 memcpy
    if (world_rank != 0) {
        void *remote_buffer;
        int remote_rank = 0;
        kupl_shm_win_query(win, remote_rank, &remote_buffer);
        memcpy(local_buffer, remote_buffer, buf_size);

        // 检查是否成功 memcpy
        bool check = true;
        for (int i = 0; i < count; i++) {
            if (((int *)local_buffer)[i] != i) {
                check = false;
                break;
            }
        }
        if (check) {
            printf("rank[%d] memcpy success\n", world_rank);
        } else {
            printf("rank[%d] memcpy fail\n", world_rank);
        }
    }
    
    // 销毁共享内存
    kupl_shm_win_free(win);
    // 销毁 kupl 通信域
    kupl_shm_comm_destroy(kupl_comm);
    MPI_Finalize();
}