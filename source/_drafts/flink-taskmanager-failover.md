---
title: Flink 局部故障快速恢复
date: 2022-05-20 21:55:34
categories: flink
comments: false
tags:
  - flink
  - 分布式计算
  - 大数据
---

Flink 当前的 Task 故障恢复策略在大部分情况下，都会重启 Job 的所有 Task，牵一发动全身，故障恢复时间较长，但是在实时推荐场景下，业务更关注实时性，偶尔丢点数据是可以接受的，如果偶发任务抖动就重启整个 Job，会导致故障恢复期间断流，无法保证实时性。事实上，一般在生产集群中，遇到的任务抖动主要是节点宕机、硬件故障以及网络抖动引发的 TaskManager 失联，为此，我们实现了一种新的 Flink 故障恢复策略，当 TaskManager 故障时，保证其他正常 TaskManager 上的 Tasks 继续运行，并快速批量恢复故障 TaskManager 上的 Tasks，做到故障恢复期间不断流并尽可能降低丢数据的时长。实际应用时，在不超过一个心跳时长的时间内可以快速应对 TaskManager 失联问题，可使实时推荐 7*24 小时稳定不断流。<!--more-->

## 背景

实时推荐场景下，业务主要关注实时性，可以允许偶发性的少量数据丢失，然而在实际生产集群中，不可避免出现机器宕机、硬件故障以及网络抖动等问题，进而造成 Flink 作业中断重启，如果作业的执行图本身较复杂，恢复时间会很长，可达 10min 以上，如果 Flink 内部恢复失败，则会引发整个 Application 的失败重新提交，恢复时间会拉的更长，一定程度上会影响实时推荐的效果。

Flink 虽然本身具备一定的 Task 故障恢复能力，但是其当前的 Task 故障恢复策略主要是从数据完整性角度考虑，包括如下两种。
* Region：只重启涉及失败 Task 的最小连通执行子图
* Full：重启整个执行图

其中 Region 恢复策略本意也是最小化重启影响的 Tasks，但是考虑数据完整性，它必须连带把失败 Task 的上下游一起取消重启，再借助 Checkpoint 机制进行数据回放，所以只有在执行图（ExecutionGraph）非全连通情况下才能做到局部重启，具体表现如下图所示。但是大部分业务场景下，一个 Job 的执行图是全连通的，因为只要业务逻辑里有 Shuffle，那就必然会使得执行图全连通，如下图最右边示例，这就退化成 Full 恢复策略了，实际表现就是牵一发动全身，故障恢复时间较长，监控曲线出现断流掉 0 现象。

<img src="/images/region-failover.png" width="600" height="400" alt="region-failover" align=center />

针对该问题，业界其实也有一些讨论及方案，社区有一个重大特性提案 [FLIP-135](https://cwiki.apache.org/confluence/display/FLINK/FLIP-135+Approximate+Task-Local+Recovery) 就是尝试做有损的故障局部恢复，从2020年8月提出，到现在也没有太大进展，看社区规划并不打算继续推进。

为此，我们需要探索自己的局部故障恢复策略，但是不一样的是，我们把故障粒度限定为 TaskManager 故障，因为几乎所有外部抖动（节点宕机、硬件故障、网络抖动等）的直接后果就是 TaskManager 失联，对于个别的 Task 失败，没必要专门处理（交给默认的 Region 恢复策略处理），这通常是因为业务逻辑没考虑全面导致，总结来说就是我们提供一种故障恢复策略，可以帮助用户去应对“天灾”，故障期间会丢点数据，但可以保证 Flink Pipeline 运行持续进行，并快速恢复。

## 现有的故障恢复介绍

在详细阐述我们实现的 TaskManager 故障恢复策略之前，有必要先介绍下 Flink 自带的故障恢复机制。

### 控制层面 - Task 故障恢复策略

控制层面主要在 JobManager 上完成，包括如下几步：
 * 感知故障
 * 计算受影响的 Tasks
 * 重启（先取消运行再重新下发）受影响的 Tasks

为了更形象地表述，下面以一个简单的 MapReduce 示例展开，如下图所示，Flink 作业有两个算子，分别做 Map 和 Reduce，并行度均为 4，跑在两个 TaskManager 上，每个TaskManager 2 个 Slot。

<img src="/images/flink-mapreduce-demo.png" width="600" height="400" alt="flink-mapreduce-demo" align=center />

感知故障分为两种情况：
1. 如果 TaskManager 进程未退出，但是其上面运行的 Task 异常，则由 TaskManager 主动上报 Task 异常信息给 JobManager，进而触发故障恢复。
2. 如果 TaskManager 进程意外退出了，那么上图中的控制通道（橙色实线）与数据通道（黑色虚线）都断了，控制通道断可导致 JobManager 与 TaskManager 之间的心跳异常，数据通道断可导致运行在其他 TaskManager 上的 Tasks 因通信异常而失败，进而上报给 JobManager ，这两种异常哪个先触达到 JobManager 就先触发故障恢复。

感知到故障后，随即按故障恢复策略计算受影响的 Tasks：
* 如果是 Full 策略，重启整个Job，受影响的 Tasks 就是 Job 中所有的 Tasks。
* 如果是 Region 策略，则以故障 Task 为基准，在执行图中找到包含故障 Task 的最小连通子图，像上图中的执行图，因为有 Shuffle，执行图是全连通的，所以计算出的受影响的 Tasks 就是 Job 中所有的 Tasks，相当于退化成 Full 策略了。

计算完受影响的 Tasks 后，先取消正在运行的 Tasks，然后通过一个异步操作重新下发这些受影响的 Tasks，下发阶段包括：
* 申请所需的 Slot 资源（JobManager 向 ResourceManager 申请，ResourceManager 必要时向集群提供者申请新的 TaskManager）。
* 按照执行图的拓扑排序逐个下发 Task。

> Q: 如果多个 Task 故障同时被上报到 JobManager 怎么办？
> A: 不用担心，JobManager 的 RPC 调用是单线程串行执行的，即使有多个 Task 故障上报，JobManager 实际会串行处理，当处理完一个接着处理下一个时，在一次故障恢复处理中，可能会发现，上一个故障已经连带处理了当前故障，所以可以不用处理。

### 执行层面 - Task 底层通信模型

在上一小节中提到，当 TaskManager 进程意外退出时，JobManager 可以通过失败的 Task 的上下游来感知故障，因为数据通道断了，势必造成上下游的 Task 通信异常，也就是说 Task 故障实际会有传递性的，如果不做处理，会引发连锁反应，在详细阐述故障时如何保持住上下游前，我们先了解下 Flink 当前执行层面的 Task 底层通信模型。

上述 MapReduce 示例的数据通信如下图所示，熟悉 Hadoop 或 Spark 的同学应该很好理解，Map 会按 key 对数据进行分组写入，Reduce 会聚合读取上游所有 Map 输出的属于同一个组的数据，在 Flink 流计算模型下，Map 和 Reduce 同时运行，数据以流的方式进行传输，Map 往 ResultPartition 写入数据，每个 ResultPartition 按 key 分组分为多个 ResultSubpartition，Reduce 从 InputGate 读取数据，每个 InputGate 包含多个 InputChannel，每个 InputChannel 聚合来自上游所有 Map 输出的同一个分组的 ResultSubpartition。

<img src="/images/flink-mapreduce-data-transfer-demo.png" width="600" height="400" alt="flink-mapreduce-data-transfer-demo" align=center />

如果上下游 Task 在同一个 TaskManager 上运行，数据传输实际是通过本地队列实现，上图用黑色实线表示，如果上下游 Task 在不同的 TaskManager 上运行，数据则需要通过网络传输，上图用黑色虚线表示，更详细地，如下图所示，Map-2 和 Reduce-3 两个 Task 通过网络进行数据流式传输，底层网络通信通过 Netty 实现，上游作为 Netty 服务端，下游作为 Netty 客户端，建立数据通道时，下游向上游发起请求，告诉上游要请求的 ResultSubpartition，上游会创建一个 SubpartitionView 注册到对应的 Netty Handler 里，当 ResultSubpartition 有数据写入时，会通知对应的 SubpartitionView 读取，进而通过 Netty Handler 把数据打包带上 InputChannel 信息发到网络，下游客户端收到数据后会分拣到对应的 InputChannel 中，如此源源不断，实现流式数据传输。

<img src="/images/flink-mapreduce-network-data-transfer-demo.png" width="600" height="400" alt="flink-mapreduce-network-data-transfer-demo" align=center />

很显然，如果 TaskManager 异常退出，那么 Netty TCP 连接会被中断，无论是上游还是下游断开，都会引发对方失败。下面重点阐述我们实现的 TaskManager 故障恢复策略。

## 总体思路

TaskManager 故障恢复策略，目的是在有 TaskManager 异常退出情况下，保证其他正常 TaskManager 上的 Tasks 继续运行，并快速批量恢复故障 TaskManager 上的 Tasks，做到故障恢复期间不断流并尽可能降低丢数据的时长，具体表现大致如下图所示。

<img src="/images/taskmanager-failover-overview.png" width="600" height="400" alt="taskmanager-failover-overview" align=center />

要做到这个效果，需要分别在执行层面和控制层面实现如下几个功能：
1. 执行层面
    * 有 tm 故障时，flink pipeline 其余部分正常工作，保证不引发连锁反应
    * 失败 tasks 被重新下发运行后，flink pipeline 可以重新接上，并恢复原样
2. 控制层面
    * 感知 TaskManager 异常，重新调度异常 TaskManager 上的 Tasks

### 执行层面实现

当 TaskManager 故障时，其上面运行的所有 Tasks 会失败，之前通过网络与这些失败 Tasks 连接的上下游 Tasks 会被影响，所以具体实现分为两种情况：1）上游 Task 故障，下游 Task 保持与恢复；2）下游 Task 故障，上游 Task 保持与恢复。

#### 上游失败，下游保持与恢复

如果是上游失败了，下游会感知到网络连接断开，如下图所示，Flink 会将网络连接异常递交给对应的 InputChannel，当 InputGate 从 InputChannel 拿数据时，会拿到异常进而触发失败。

<img src="/images/upstream-fail.png" width="600" height="400" alt="upstream-fail" align=center />

为了保持住下游，我们实现了自己的 InputChannel，我们命名为 SuspendableRemoteInputChannel，当被递交网络连接异常时，它会直接清空其队列中的所有数据，并置为暂停状态，InputGate 不会从暂停状态的 InputChannel 拿数据。如果上游失败的 Task 被重启了，暂停状态的 InputChannel 需要重新恢复连接，如下图所示，JobManager 重新下发上游 Task 后，等其上报运行状态后，由 JobManager 去通知下游，告诉下游 Task 相关新的连接信息，下游 Task 根据新的连接信息，向新的上游发起 subpartition 请求建立数据传输通道，恢复暂停状态的 InputChannel。

<img src="/images/upstream-fail-downstream-resume.png" width="600" height="400" alt="upstream-fail-downstream-resume" align=center />

#### 下游失败，上游保持与恢复

如果是下游失败了，上游会感知到网络连接断开，如下图所示，Flink 会将对应的 SubpartitionView 关闭（SubpartitionView 负责读取 Subpartition 的数据），没有了读，ResultSubpartition 中的队列很快会被写满，进而会导致反压。

<img src="/images/downstream-fail.png" width="600" height="400" alt="downstream-fail" align=center />

为了保持上游的健康，我们实现了自己的 ResultSubpartition 和 SubpartitionView ，我们命名为 SuspendablePipelineSubpartition 和 SuspendablePipelineSubpartitionView，当上游感知到网络连接断开时， SuspendablePipelineSubpartitionView 会被关闭，它会进一步去清空 SuspendablePipelineSubpartition 中的队列，并将其置为暂停状态，处于暂停状态的 Subpartition 在被写入数据时会直接丢弃，相当于这期间这一路 Subpartition 是在丢数据的，不然会导致反压。如果下游失败的 Task 被重启了，暂停状态的 Subpartition 恢复写入，如下图所示，因为重新下发下游 Task 时，Task 本身就知道其上游的连接信息，所以不需要 JobManager 介入控制，下游 Task 运行起来后，会主动去连接上游，发起 Subpartition 请求，上游收到请求后，会新建一个 SubpartitionView 并恢复暂停状态的 Subpartition，Subpartition 就会被正常写入数据，SubpartitionView 正常读取。

<img src="/images/downstream-fail-upstream-resume.png" width="600" height="400" alt="downstream-fail-upstream-resume" align=center />

### 控制层面实现

前面提到，当故障时，可以实现上下游不引发连锁反应，恢复还是要靠控制层面这个大脑，控制层面 JobManager 正常情况下在调度下发 Task 前，会从 ResourceManager 申请 Slot 资源，如果资源不足，ResourceManager 会从 Cluster Provider 申请 TaskManager。我们的 Flink 作业是通过 Flink-K8S-Operator 部署到 K8S 上，详情请参考[前面的文章](https://km.woa.com/group/24938/articles/show/457927)，如果 TaskManager 异常退出了，K8S 会自动拉起一个新的 Pod 重新运行 TaskManager。所以 JobManager 上的逻辑较简单，只需要负责重新调度故障 TaskManager 上的 Tasks，资源申请逻辑是现成的。具体实现包括两块：
* 故障感知：JM 感知 TM 心跳是否异常
* 异常处理：计算失败的 Tasks，按拓扑排序顺序重新下发

<img src="/images/taskmanager-failover-strategy.png" width="600" height="400" alt="taskmanager-failover-strategy" align=center />

当 JobManager 感知到有 TaskManager 心跳异常时，将异常包装成我们扩展的 TaskManagerLostException，紧接着会进入故障恢复逻辑处理，我们新增了一个 FailoverStrategy 的实现类 RestartTaskManagerFailoverStrategy，它专门针对TaskManagerLostException 类型的异常，只处理异常 TaskManager 上的 Tasks，其他异常，则会退化交给默认的 region 恢复策略处理。

### 若干细节问题处理

#### 网络分区问题

因为我们是通过网络异常来感知 TaskManager 故障，所以必须考虑网络分区问题，如果 JM 跟 TM 断了，实际 TM 是正常运行的，执行层面的 flink pipeline 也正常，如下图左边所示，这时 JM 会认为 TM 异常了，触发故障恢复，重新拉起对应 Tasks，其实这不影响，重新拉起 Tasks 会替换原来的 Tasks 接到执行 pipeline 中。

<img src="/images/networke-partition.png" width="600" height="400" alt="networke-partition" align=center />

如果 TM 与 TM 之间断了，如上图右边所示，因为我们会在网络异常时保持住 Task 不让其异常，只是暂停对应的 InputChannel 或 ResultSubpartition，这就会导致有些数据通道丢数据，而 JM 却没感知，针对这种情况，我们在 InputChannel 或 ResultSubpartition 进入暂停状态时，开启一个定时器，如果指定时间内没有恢复，则直接失败，进而触发默认的 region 恢复策略。

#### 多个 TaskManager 同时故障问题

一般单机故障可能导致一个 Flink Application 的多个 TM 退出，如何处理多个 TaskManager 跪掉的情况呢？实际不用过于担心，JobManager 感知 TaskManager 心跳异常是串行的，所以串行触发故障恢复策略即可。但是有个细节需要处理，当针对第一个 TaskManager 故障恢复时，在下发 Task 前，可能无法决定其上游网络连接信息而失败，这时我们直接给个错误连接信息，当实际下发执行时，会因为连接失败进入暂停状态，等上游被下发执行后，它会被指示更新连接信息而恢复。

## 小结

本文介绍了在适应实时推荐业务场景下实现的 TaskManager 故障恢复策略，目的是应对外部节点宕机、硬件故障、网络抖动等引发的 TaskManager 异常退出问题，做到异常时不断流，并快速恢复，保证实时性。除了“天灾”以外，顺带也可以解决一些“人祸”，例如用户业务逻辑里有内存泄漏，导致 TaskManager 运行一段时间因为 oom 被 kill 等问题。
