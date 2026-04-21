#### 介绍

`hpckit-sample`旨在提供HPCkit各组件相关的使用用例

#### 软件架构

一级目录分为组件目录和综合用例目录，每个组件目录下有多个小目录，每个小目录中有对应的用例（源码文件，makefile 文件，readme说明）

comprehensive-example：综合用例

kupl: 鲲鹏统一并行加速库相关用例

#### 安装教程

1.  安装最新的 HPCKit

    下载地址：https://www.hikunpeng.com/developer/hpc/hpckit-download
    
    安装指南：https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/instg/topic_0000001806090516.html
    
    开发指南：https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hpckit/devg/KunpengHPCKit_developer_003.html

2.  下载本文件
3.  用例编译

    (1) 一键式全量编译安装：

        在根目录下执行 `sh build.sh`

    (2) 按需编译安装：

        进入到具体用例所在目录下根据 readme 的说明 进行编译和运行

#### 使用说明

1.  本用例仅供参考，具体使用方法请看 HPCKit 官方手册
2.  本用例提供两种不同的编译/执行方式，用户可以自由选择