---
title: Spark Cache性能测试
date: 2017-03-22  22:11:29
categories: spark
comments: false
tags:
  - spark
  - cache
  - benchmark
  - 分布式计算
---

采用Spark自带的Kmeans算法作为测试基准（Spark版本为2.1），该算法Shuffle数据量较小，对于这类迭代型任务，又需要多次加载训练数据，此测试的目的在于评判各种Cache IO的性能，并总结其Spark内部原理作分析，作为Spark用户的参考。<!--more-->

## 测试准备

训练数据是通过[Facebook SNS公开数据集生成器](http://prof.ict.ac.cn/BigDataBench/dowloads/)得到，在HDFS上大小为9.3G，100个文件，添加如下两个参数以保证所有资源全部到位后才启动task，训练时间为加载数据到训练完毕这期间的耗时。

```
--conf spark.scheduler.minRegisteredResourcesRatio=1
--conf spark.scheduler.maxRegisteredResourcesWaitingTime=100000000
```

测试集群为3个节点的TS5机器搭建而成，其中一台作为RM，并运行着Alluxio Master，两个NM上同时运行着Alluxio Worker。除以上配置外，其他配置全部保持Spark默认状态。公共资源配置、分区设置以及算法参数如下表所示，executor_memory视不同的测试用例不同:

| driver_memory | num_executor | executor_cores | 分区数 |  聚类个数 |  迭代次数 |
| :--------: | :--------:| :--------:| :--------:| :--------:| :--------:|
| 4g | 10 | 2 | 100 | 6 | 10 |


## 测试用例

### 测试用例1:  不使用Cache

在不使用Cache的情况下，测试Spark-Kmeans算法的训练时间以及GC时间占比。这里分别使用两种方式加载数据，一种是直接从HDFS加载数据，另一种是透过ALLUXIO加载数据，相关测试指标数据如下表所示：

| \ | 说明 | 内存使用总量 | 训练时间 | GC时间占比 |
| :-------: | :--------: | :--------:| :--------:| :--------:|
| case 1-1 | 从hdfs直接加载训练数据 | 20g | 1064s | 3.1% |
| case 1-2 | 透过alluxio加载训练数据 | 20g + 9.3g | 689s | 3.4% |

不使用cache时，以上两种情形GC均不是瓶颈，主要差别表现在：
* 从hdfs直接加载训练数据：在每次迭代时均要读一遍hdfs，访问hdfs有较大的开销；
* 透过alluxio加载训练数据：只需第一次加载读一遍hdfs，后续迭代直接从alluxio中读取，不过alluxio额外消耗9.3G内存，整体性能提升35%+。

### 测试用例2:  使用Cache

在使用Cache的情况下，从HDFS加载数据后先做cache，分别采用不同的Cache方式，相关测试指标数据如下表所示：

| \ | 缓存方式 | executor_memory | 内存使用总量 | cache比例 | 训练时间 | GC时间占比 |
| :----: | :----: | :----: | :----: | :----:| :----: | :----:|
| case 2-1 | MEMORY_ONLY | 2g | 20g | 33% | 1558s | 12% |
| case 2-2 | MEMORY_ONLY | 4g | 40g | 90% | 986s | 7% |
| case 2-3 | MEMORY_ONLY | 6g | 60g | 100% | 463s | 4.7% |
| case 2-4 | MEMORY_AND_DISK | 2g | 20g | 100% | 1182s | 16.9% |
| case 2-5 | DISK_ONLY | 2g | 20g | 100% | 514s | 3.2% |
| case 2-6 | ALLUXIO | 2g | 20g + 9.3g | 100% | 687s | 4.5% |

> [采用Alluxio的Cache实现方式](https://alluxio.com/blog/alluxiospark-rdd)为:
> ```scala
data.saveAsTextFile(path)
val data = sc.textFile(path)
```

从以上测试数据看来，让人有点出乎意料，一开始有点不太相信，但是多次测试后数据并没有多大的抖动，所以说Spark的性能受多方面因素的影响，单单Cache这块不同的Cache方式以及不同的资源情况下，其性能差别就相差较大，下面分析其内在原因。

从HDFS加载训练数据后直接采用Spark原生的Cache，当executor_memory为2g时，不足以Cache住原始训练数据，从UI上看到Cache的比例只有33%左右，导致频繁的 rdd-block 剔除重建，同时由于内存吃紧，可能引发频发的Spill以及较重的GC，从UI上看到GC时间占到总的task运行时间的12%左右，已经成为瓶颈，其整体性能还不如不使用Cache的case1-1；当executor_memory为4g时，也不足以Cache住原始训练数据，但是其Cache的比例有90%左右，同样存在 rdd-block 剔除重建，频发Spill以及较重的GC，GC时间占总的task运行时间的7%左右，虽然比executor_memory为2g的情况有所好转，但是仍然不理想，只比不做Cache的case1-1好7%左右，但是内存却多用了20g，并不是特别划算；当executor_memory为6g时，可以全部Cache住原始训练数据，性能较优，GC占比较小，但是比不用Cache的case1-1要多用40g内存，有些代价。

一般来说，当我们内存不够时，可以选择MEMORY_AND_DISK的缓存方式，但是测试发现MEMORY_AND_DISK的缓存效果并不是特别好，从测试数据来看，还不如直接使用DISK_ONLY的缓存方式，MEMORY_AND_DISK的缓存方式带来的GC开销非常大，可能是因为每次都尽可能地Cache数据到内存，不够再刷到磁盘，造成JVM频繁GC。

另外测试了使用Alluxio作缓存的Case，发现并没有[官方描述](https://alluxio.com/blog/alluxiospark-rdd)的那样会提升Cache的性能，还不如直接使用Spark DISK_ONLY缓存，感觉官方给的测试对比数据存在一定的水分，值得一提的是在多个Application之间Alluxio能起到加速作用。


## 小结

Spark的Cache并不是总是会加速任务运行，Cache的方式不同，对任务产生的影响不同。并不是能用内存Cache就用内存，而是要考虑是否有充足的内存Cache住你的数据，否则可能适得其反。
