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
make commit // 提交任务到任务机上运行，主要用于提交评分，会运行cpu test
make test // 提交任务到任务机上运行，主要用于测试，不会运行cpu test。
make prof //提交任务到任务机，并使用hipprof分析程序
make dump // 反汇编
```

## 版本历史
### v1.1
- 修改了im2col的展开方法，采用2D展开，提高了L2 cache的命中率。

|样例|耗时(us)|
|---|---|
|preliminary_1| 823.6|
|preliminary_2| 2210.6|
|preliminary_3| 1189.5|
|preliminary_4| 442.5|
|preliminary_5| 1327.1|
|preliminary_6| 1441.8|

### v1.0
- 基于赛方的例子实现了gemm_32x32x16_fp。
- 使用im2col算法，采取列展开。

|样例|耗时(us)|
|---|---|
|preliminary_1| 878.12|
|preliminary_2| 4217.9|
|preliminary_3| 1173.1|
|preliminary_4| 438.5|
|preliminary_5| 1331.1|
|preliminary_6| 1443.3|

## 参考资料
...
