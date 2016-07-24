---
title: 深入理解Spark(二)：Spark Scheduler原理及资源配置
date: 2016-07-23  21:33:14
categories: spark
tags:
  - spark
  - Scheduler
  - 大数据
  - 分布式计算
---

通过文章[“Spark核心概念RDD”](/posts/spark-rdd.html)我们知道，Spark的核心是根据RDD来实现的，Spark Scheduler则为Spark核心实现的重要一环，其作用就是任务调度。Spark的任务调度就是如何组织任务去处理RDD中每个分区的数据，根据RDD的依赖关系构建DAG，基于DAG划分Stage，将每个Stage中的任务发到指定节点运行。基于Spark的任务调度原理，我们可以合理规划资源利用，做到尽可能用最少的资源高效地完成任务计算。<!--more-->

### 分布式运行框架

Spark可以部署在多种资源管理平台，例如Yarn、Mesos等，Spark本身也实现了一个简易的资源管理机制，称之为Standalone模式。由于工作中接触较多的是Saprk on Yarn，不做特别说明，以下所述均表示spark-on-yarn。Spark部署在Yarn上有两种运行模式，分别为yarn-client和yarn-cluster模式，它们的区别仅仅在于Spark Driver是运行在Client端还是ApplicationMater端。如下图所示为Spark部署在Yarn上，以yarn－cluster模式运行的分布式计算框架。

<img src="/images/spark-distribution-framework.png" width="400" height="230" alt="spark-distribution-framework" align=center />

其中蓝色部分是Spark里的概念，包括Driver和Executor，Driver负责分发任务以及监控任务运行状态，Executor负责执行任务，Executor是一个进程，运行在其中的任务是线程，所以说Spark的任务是线程级别的。当以yarn－cluster模式运行Spark应用程序时，Driver运行在NodeManager上，作为ApplicationMaster角色，当Driver启动后，会根据提交时申请的资源参数，向ResourceManager申请资源运行Executor，一个Executor对应与一个Container，Executor进程起来后，等待Driver分发任务并将任务执行状态上报给Driver。

### Spark任务调度总览

当Driver和Executor起来后，Driver则会根据用户程序逻辑分发任务。在详细阐述任务调度前，首先说明下Spark里的几个概念。一个Spark应用程序包括Job、Stage以及Task三个概念：
* Job是以action方法为界，遇到一个action方法则触发一个Job；
* Stage是Job的子集，以RDD宽依赖(即shuffle)为界，遇到shuffle做一次划分；
* Task是Stage的子集，以并行度(分区数)来衡量，分区数是多少，则有多少个task。

Spark的任务调度总体来说分两路进行，一路是stage级的调度，一路是task级的调度，总体调度流程如下图所示。

<img src="/images/spark-scheduler-overview.png" width="400" height="230" alt="spark-scheduler-overview" align=center />

Spark RDD通过其transactions操作，形成了RDD血缘关系图，即DAG，最后通过action的调用，触发Job并调度执行。DAGScheduler负责stage级的调度，主要是将DAG切分成若干stages，并将每个stage打包成TaskSet交给下游TaskScheduler调度。TaskScheduler负责task级的调度，将上游DAGScheduler给过来的TaskSet按照指定的调度策略分发到指定Executor上执行，调度过程中SchedulerBackend负责提供可用资源，其中SchedulerBackend有多种实现，分别对接不同的资源管理系统。

#### stage级的调度



#### task级的调度



### 资源合理配置



### 动态资源申请
