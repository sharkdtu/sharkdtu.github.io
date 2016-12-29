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

## 分布式运行框架

Spark可以部署在多种资源管理平台，例如Yarn、Mesos等，Spark本身也实现了一个简易的资源管理机制，称之为Standalone模式。由于工作中接触较多的是Saprk on Yarn，不做特别说明，以下所述均表示spark-on-yarn。Spark部署在Yarn上有两种运行模式，分别为yarn-client和yarn-cluster模式，它们的区别仅仅在于Spark Driver是运行在Client端还是ApplicationMater端。如下图所示为Spark部署在Yarn上，以yarn－cluster模式运行的分布式计算框架。

<img src="/images/spark-distribution-framework.png" width="100%" height="100%" alt="spark-distribution-framework" align=center />

其中蓝色部分是Spark里的概念，包括Driver和Executor，Driver负责分发任务以及监控任务运行状态，Executor负责执行任务，Executor是一个进程，运行在其中的任务是线程，所以说Spark的任务是线程级别的。当以yarn－cluster模式运行Spark应用程序时，Driver运行在NodeManager上，作为ApplicationMaster角色，当Driver启动后，会根据提交时申请的资源参数，向ResourceManager申请资源运行Executor，一个Executor对应与一个Container，Executor进程起来后，等待Driver分发任务并将任务执行状态上报给Driver。

## Spark任务调度总览

当Driver和Executor起来后，Driver则会根据用户程序逻辑分发任务。在详细阐述任务调度前，首先说明下Spark里的几个概念。一个Spark应用程序包括Job、Stage以及Task三个概念：
* Job是以action方法为界，遇到一个action方法则触发一个Job；
* Stage是Job的子集，以RDD宽依赖(即shuffle)为界，遇到shuffle做一次划分；
* Task是Stage的子集，以并行度(分区数)来衡量，分区数是多少，则有多少个task。

Spark的任务调度总体来说分两路进行，一路是stage级的调度，一路是task级的调度，总体调度流程如下图所示。

<img src="/images/spark-scheduler-overview.png" width="100%" height="100%" alt="spark-scheduler-overview" align=center />

Spark RDD通过其transactions操作，形成了RDD血缘关系图，即DAG，最后通过action的调用，触发Job并调度执行。DAGScheduler负责stage级的调度，主要是将DAG切分成若干stages，并将每个stage打包成TaskSet交给下游TaskScheduler调度。TaskScheduler负责task级的调度，将上游DAGScheduler给过来的TaskSet按照指定的调度策略分发到指定Executor上执行，调度过程中SchedulerBackend负责提供可用资源，其中SchedulerBackend有多种实现，分别对接不同的资源管理系统。

### stage级的调度

Spark的任务调度是从DAG切割开始，主要是由DAGScheduler来完成。当遇到一个action操作后就会触发一个Job的计算并交给DAGScheduler来提交，它首先会根据RDD的血缘关系构成的DAG进行切分，将一个Job划分为若干Stages，具体划分策略是根据是否有宽依赖，即是否需要做shuffle来划分，以shuffle为界，划分上下游Stage，窄依赖的RDD之间被划分到同一个Stage中，可以进行pipeline式的计算，划分的Stages分两类，一类叫做ResultStage，为DAG最下游的Stage，由action决定，另一类叫做ShuffleMapStage，为下游Stage准备数据，如下图所示。

<img src="/images/spark-scheduler-dag.png" width="100%" height="100%" alt="spark-scheduler-dag" align=center />

上图中DAG有两个action，那么则会触发两个Job，根据RDD的依赖关系，union和map属于窄依赖，则划分到同一个Stage中，join属于宽依赖，则以此为界，划分上下游Stage，类似地，groupByKey和aggregate也属于宽依赖而被划分。一般来说这种情况会将中间Stage1中的RDD缓存起来以避免重计算。

Stage划分过程中会分别记录Stage与Job之间的对应关系，因为有时候多个Job会共享同一个Stage，如上图实例所示，Stage0和Stage1就被两个Job共享，如果Stage1中RDD做了缓存，在第一个Job计算过程中会记录Stage1中各partition的缓存地址信息，那么下次计算第二个Job过程中则不需要再次调度计算Stage1了。

Stage划分后，则将Stage按依赖关系依次调度执行，Stage会被打包成TaskSet交给TaskScheduler，一个partition对应一个task，并监控Stage的运行情况，只有executor丢失或者task由于Fetch失败才需要重新提交失败的Stage以调度运行失败的任务，其他类型的task失败会在TaskScheduler的调度过程中重试。

### task级的调度

Spark task调度是由TaskScheduler来完成，由前文可知，DAGScheduler将Stage打包到TaskSet交给TaskScheduler，TaskScheduler会将其封装为TaskSetManager以监控管理task的运行，TaskScheduler就是以TaskSetManager为单元来调度任务。TaskScheduler的任务调度会借助SchedulerBackend，SchedulerBackend定义了许多与Executor事件相关的处理，包括：新的Executor注册进来的时候记录Executor的信息，增加全局的资源量(核数)，进行一次makeOffer；Executor更新状态，若任务完成的话，回收core，进行一次makeOffer；其他停止Executor、删除executor等事件。makeOffer的目的是在有资源更新的情况下，会把现有的Executor资源以WorkerOfffer列表的方式传给TaskScheduler，通知TaskScheduler对现有的任务进行一次分配，最终在任务分配的Executor上启动tasks，总体流程大致如下图所示。

<img src="/images/spark-scheduler-task.png" width="100%" height="100%" alt="spark-scheduler-task" align=center />

TaskScheduler拿到这一些资源后，去遍历DAGScheduler提交过来的TaskSet并根据locality决定如何启动tasks。首先它会将所有等待提交的TaskSet进行一次优先级排序，这个排序算法目前是两种：FIFO或FAIR。得到一份待运行的TaskSet后，然后把SchedulerBackend交过来的资源信息合理分配给这些tasks，分配前，为了避免每次都是前几个Executor被分到tasks，所以先对资源列表进行一次随机洗牌，随后就是遍历tasks，根据task的locality安排任务运行所在的Executor并通过rpc调用在对应的Executor启动task。

task的locality有五种，按优先级高低排：PROCESS_LOCAL，NODE_LOCAL，NO_PREF，RACK_LOCAL，ANY。也就是最好在同个进程里，次好是同一台机器上，再次是同机架，或任意都行。每个task都有自己的locality，如果本次WorkerOfffer资源里没有想要的locality资源，则会等待一定的时间，如果超时则逐步降低locality级别调度运行。

## 资源申请机制

Spark之前只支持静态资源申请，即一开始就指定用多少资源，在整个Spark应用程序运行过程中用的资源都不变。后来支持动态Executor申请，用户不需要指定Executor的数量，Spark会动态调整Executor的数量以达到资源利用的最大化。

### 静态资源申请

静态资源申请是一开始就要估计应用程序需要使用的资源，包括Executor数(num_executor)、每个Executor上的core数(executor_cores)、每个Executor的内存(executor_memory)以及Driver的内存(driver_memory)，如果Spark是部署在Yarn上，当ApplicationMaster启动后，会单独启动一个线程不断地申请指定数量的Executor，前面也说到，Executor更新信息会同步到SchedulerBackend。这些资源的使用与任务的并行度(Parallelism)有关，它们在任务运行时具体表现如下图所示。

<img src="/images/spark-scheduler-resource.png" width="100%" height="100%" alt="spark-scheduler-resource" align=center />

任务的并行度(Parallelism)由分区数决定，有多少的分区，就会有多少的task。每个task占用一个core，一个Executor上的所有core共享Executor上的内存，一次并行运行的task数等于num_executor*executor_cores，如果分区数超过该值，则需要运行多个轮次，一般来说建议运行3～5轮较为合适，否则考虑增加num_executor或executor_cores。由于一个Executor的所有tasks会共享内存executor_memory，所以建议executor_cores不宜过大。executor_memory的设置则需要综合每个分区的数据量以及是否有缓存等逻辑。

### 动态资源申请

动态资源申请目前只支持到Executor，即可以不用指定num_executor，通过参数spark.dynamicAllocation.enabled来控制，其他的还是需要自己来规划的。由于许多Spark应用程序一开始可能不需要那么多Executor或者其本身就不需要太多Executor，所以不必一次性申请那么多Executor，根据具体的任务数动态调整Executor的数量，尽可能做到资源的不浪费。由于动态Executor的调整会导致Executor动态的添加与删除，如果删除Executor，其上面的中间shuffle结果可能会丢失，这就需要借助第三方的ShuffleService了，如果Spark是部署在Yarn上，则可以在Yarn上配置Spark的ShuffleService，具体操作仅需做两点:
1. 首先在yarn-site.xml中加上如下配置：
```xml
<property>
  <name>yarn.nodemanager.aux-services</name>
  <value>mapreduce_shuffle,spark_shuffle</value>
</property>
<property>
  <name>yarn.nodemanager.aux-services.spark_shuffle.class</name>
  <value>org.apache.spark.network.yarn.YarnShuffleService</value>
</property>
<property>
  <name>spark.shuffle.service.port</name>
  <value>7337</value>
</property>
```
2. 将Spark ShuffleService jar包\$SPARK_HOME/lib/spark-\*-yarn-shuffle.jar拷贝到每台NodeManager的\$HADOOP_HOME/share/hadoop/yarn/lib/下，并重启所有的NodeManager。

## 小结

本文详细阐述了Spark的任务调度与资源规划等，着重讨论Spark on Yarn的部署调度，明白Spark的任务调度原理有助于优化应用程序逻辑以及合理优化资源利用等，后续文章将逐步介绍Spark的运行原理，包括Shuffle、Storage、RPC等等。
