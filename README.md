# 江南大学-T202410295993897-DeepOptimized
这个仓库存放的是此次比赛相关的代码。

## TODO
- [x] 基于mma指令实现gemm_32x32x16_fp16
- [x] 使用gemm_32x32x16_fp16实现了基于im2col算法的conv2d
- [x] 基于implicitgemm算法的conv2d
- [ ] Swizzle机制优化L2命中率
- [ ] im2col混合展开优化L2命中率
- [ ] 使用mma指令实现implicitgemm

## 使用方法
```
make job // 提交任务到任务机上运行
make prof //提交任务到任务机，并使用hipprof分析程序
make dump // 反汇编
```

## 参考资料
...
