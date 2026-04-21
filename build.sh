# 自动编译全部 demo 脚本

# 工程目录
PROJECT_DIR="./"

# 递归查找所有包含 makefile 的目录
find "$PROJECT_DIR" -type f -name "makefile" | while read makefile; do

    # 获取 makefile 的目录
    dir=$(dirname "$makefile")

    # 进入目录
    cd "$dir" || continue

    # 执行 make
    make

    # 返回上级目录且不打印信息
    cd - > /dev/null

done

echo “所有 demo 已编译完成”