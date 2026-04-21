arm intrinsic实现的矩阵算力加速int8_gemm_bf16_dequant融合算子用例

用例描述: 输入矩阵 $A\in\mathbb{R}^{M\times K}, B\in\mathbb{R}^{K\times N}$,反量化系数 $p\in\mathbb{R}^{M},q\in\mathbb{R}^{N}$,计算结果矩阵 $C_{i,j}=\sum_{k=0}^{K-1}A_{i,k}\cdot B_{k,j}\cdot p_i\cdot q_j$。其中输入矩阵$A,B$的数据类型为int8, 反量化系数$p,q$和结果矩阵$C$的数据类型为bf16。


编译命令: make clean & make 

运行命令: 
    1. 通过make执行用例: make run
    2. 通过二进制int8_gemm_bf16_dequant执行测试用例, 三个参数依次是矩阵规模M、N、K: ./int8_gemm_bf16_dequant 128 7168 128 (K必须是4的倍数)

清理命令: make clean