---
title: 借助Spark调度MPI作业
date: 2018-11-04  15:21:44
categories: spark
comments: false
tags:
  - spark
  - mpi
  - 大数据
  - 分布式计算
---

在Spark-2.4.0中社区提出一种新的Spark调度执行方式，名为[Barrier Execution Mode](https://jira.apache.org/jira/browse/SPARK-24374)，旨在通过Spark去调度分布式ML/DL(机器学习/深度学习)训练作业，这种训练作业一般是通过其他框架实现，例如MPI、TensorFlow等。由于Spark本身的计算框架是遵循MapReduce架构的，所以在调度执行时，每个Task都是独立执行的。然而，MPI这类分布式计算作业运行时需要所有Task一起执行，Task与Task之间需要相互通信。为了将诸如MPI分布式机器学习作业很好地与Spark结合，Spark社区提出一种新的Barrier Scheduler，可保证所有的Task全部调起(当然要保证资源足够)。<!--more-->

## API风格介绍
众所周知，Spark其实本质上仍然是MapReduce架构，它能颠覆Hadoop的很重要一点是它提供了极其易用的API，用户不再需要绞尽脑汁将问题抽象成Map和Reduce，而是直接通过Spark提供的一个个算子来组织业务逻辑。新增Barrier Execution Mode当然不能破坏Spark本身的易用性，沿用原有的RDD API风格，例如调起一个MPI作业的大致轮廓为：
```scala
rdd.barrier().mapPartitions { (iter, context) =>
  // Maybe write iter to disk.
  ???
  // Wait until all tasks finished writing.
  context.barrier()
  // The 0-th task launches an MPI job.
  if (context.partitionId() == 0) {
    val hosts = context.getTaskInfos().map(_.host)
    // Set up MPI machine file using host infos.
    ???
    // Launch the MPI job by calling mpirun.
    ???
  }
  // Wait until the MPI job finished.
  context.barrier()
  // Collect output and return.
  ???
}
```

上述代码首先将数据写到本地磁盘，以供mpi程序读取，在正式启动mpi程序前，先通过`barrier`操作同步等待所有Task完成前序工作，然后通过第一个Task(一般为rank=0的mpi进程)去拉起一个mpi job，拉起mpi job的参数通过`context`相关方法获取，最后等待所有mpi任务执行完毕，mpi任务将结果写到本地磁盘，由spark最后完成结果收集。

## 小结
如今大数据平台或机器学习平台一般都会提供Spark作业的调度，而对mpi作业的支持可能没有统一的调度方式，如果你的作业恰好需要mpi来实现，却没有一个平台来支持，这个时候我们可能会给平台方提需求，要求其支持你的mpi作业调度，但是这就拉长了你的开发周期，很多东西变得不可控，通过上述spark拉起mpi作业的方式，你可以将你的mpi作业变成一个spark作业，快速完成部署。另一方面，mpi作业的输入数据一般是要提前预处理好，这部分工作spark完全可以胜任，这样就可以通过pipeline的方式将整个业务逻辑串起来。
