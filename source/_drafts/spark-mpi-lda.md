---
title: WLDA--Spark与MPI碰撞的火花
date: 2019-01-05  16:01:23
categories: spark
comments: false
tags:
  - spark
  - mpi
  - 大数据
  - 分布式计算
---

恰逢Spark-2.4.0发布不久之际，接到LDA算法实现需求，经过一番了解，算法同学反馈SparkLDA极其的慢，而且非常占资源，虽然业界也有各种各样的实现，但是很难跟现有的平台融合起来，始终找不到一个好用的版本，为此，我们开始着手自行开发实现LDA。Spark目前对LDA有两种实现，一种是基于GraphX实现的EM算法，使用Gibbs采样，但是这种实现训练速度慢，同时极其耗费内存，另一种是基于在线变分贝叶斯实现的Online LDA算法，训练速度相对于EM算法快，但是它的模型是全部存放在Driver上，每轮迭代需要广播，如果模型较大，性能会极度恶化。自行开发的话需要同时兼顾性能、易用性，同时要跟现有的平台完美融合，不能专门发明新的玩法，否则也只能用于吹牛。在Review Spark的新特性中发现，Spark-2.4.0版本提出一种新的调度方式[Barrier Scheduling](https://jira.apache.org/jira/browse/SPARK-24374)，通过该特性可以很好地将Spark与MPI融合，所以我们想到可以尝鲜这种新特性，通过MPI实现LDA核心算法，再通过Spark包装提升易用性，可以完美地解决用户的痛点。虽然Spark的Barrier Scheduling特性还处于实验阶段，但是总体上符合我们的需求，通过一些扩展，我们基于Spark和MPI开发了WLDA，性能上可以大幅超越SparkLDA，同时易用性方面可以与SparkLDA媲美。<!--more-->


## Spark与MPI的恩怨情仇

早在Spark问世以前，分布式机器学习领域始终没出现一个大家都认可的框架，基本都是各玩各的，有使用Mahout的，有使用MPI的，等等，但是这些框架或接口要么是性能太差，要么是易用性太差，例如Mahout，开始是基于MapReduce实现，可以很容易提交到Hadoop集群，但是其性能始终是个痛点，基于MPI自己撸的话，使用灵活性又很差，跟Hadoop生态很难融合到一起。Spark的横空出世打破了这种尴尬的局面，其MLlib在易用性、以及性能方面都做到用户较为满意的一个程度，MPI的易用性和灵活性曾一度被Spark拥护者们痛批，然而近年来由于数据量的爆发以及深度学习的风靡，Spark MLlib大有被各种深度学习框架淘汰的趋势，让Spark哭笑不得的是，各深度学习框架，如TensorFlow、PyTorch等都开始拥抱MPI，为了求生存，Spark也开始不得不打出AI的口号，支持MPI，融合深度学习框架，做Spark+AI的拓展，最近Spark-2.4.0版本推出新的调度方式[Barrier Execution Mode](https://jira.apache.org/jira/browse/SPARK-24374)就是为了支持MPI调度（详细解读请参考我的[另一篇博客](http://sharkdtu.com/posts/spark-mpi.html)）。

## 总体流程

根据Spark Barrier Scheduling思路，我们可以在Task中拉起MPI进程，每个MPI进程处理单个Partition的数据，这样Spark就将每个Task的工作委托给MPI进程来完成，整体分布式运行架构如下图所示。

<img src="/images/spark-mpi-lda-overview.png" width="500" height="400" alt="spark-mpi-lda-overview" align=center />

实际运行过程中，Spark Task（JVM）与MPI Process（C++ Native）之间的时序关系如下图所示。

<img src="/images/spark-mpi-lda-process-sequence.png" width="500" height="400" alt="spark-mpi-lda-process-sequence.png" align=center />

在Spark Task拉起MPI-Process后，首先会给MPI-Process喂数据，喂完数据后，一直等待MPI-Process的输出；MPI-Process加载完Spark Task喂过来的数据后，开始迭代训练，训练完毕后，将输出传回给Spark Task，并结束退出；Spark-Task获取MPI-Process输出后可以继续其他工作。

## MPI核心实现

为了对标Spark，WLDA核心是参考Spark LDA EM算法来实现的，分布式运行基础是基于MPI实现，使用openmp并行加速，如下图所示，MPI进程分为2组：word进程和document进程。

<img src="/images/spark-mpi-lda-mpi-processes.png" width="500" height="400" alt="spark-mpi-lda-mpi-processes.png" align=center />

### Word进程

Word进程是用来存储WLDA的模型，每个进程只负责模型的一部分，模型均匀分布在不同的word进程中。Word进程负责为doc进程提供word-topic模型参数，响应doc进程发送过来的模型更新以及拉取消息。其中WLDA的模型指的是word-topic矩阵（矩阵大小 = 词汇数目 x 主题数目），矩阵每行表示语料库中的一个词在各个topic中的权重，由于LDA算法本身算是一种无监督算法，一般训练完毕后模型只是一个产物。

### Document进程

Doc进程是用来训练WLDA文档的，采用数据并行，每个进程只持有语料库的一部分文档，同时维护训练过程中该部分文档的主题分布，即doc-topic矩阵（矩阵大小 = 文档数目 x 主题数目），也就是我们需要得到的结果。doc进程从word进程获取word-topic参数和global_topic参数（由word-topic矩阵按行叠加），根据Gibbs Sampling算法为每个词的重新选取主题，将词的主题选取情况发送消息给word进程，通知其更新对应的模型参数。

### 训练过程

整个分布式训练过程描述如下：

1. 初始化输入
    * Doc进程读取文档输入，每行一个文档，与Spark兼容，libsvm格式，(docId wordid:wordcount ...)
    * Word进程启动模型服务，开始时word-topic矩阵均为0
2. 随机初始化word-topic矩阵和doc-topic矩阵
    * Doc进程随机为每个词选取主题，更新本地的doc-topic矩阵，以及通过MPI消息通知word进程更新word-topic矩阵
    * Word进程等待doc进程推过来的word-topic矩阵，更新模型参数
3. 训练迭代
    * Doc进程从word进程拉取global_topic参数（由word-topic矩阵按行叠加），使用Gibbs Sampling算法对本地每个词重新选取主题，更新doc-topic矩阵，并通过MPI消息通知word进程更新word-topic矩阵
    * Word进程等待doc进程拉取word-topic参数和global_topic参数、以及更新word-topic矩阵
4. 训练完毕输出
    * Doc进程输出doc-topic矩阵
    * Word进程输出word-topic矩阵

## Spark包装

上述MPI实现实际上是很难提交到我们现有的平台，例如需要提交Hadoop Yarn集群运行，首先需要基于Yarn的API写一个分布式程序来驱动，另外需要在MPI程序中支持不同文件系统的访问，这么一来需要连带的开发量特别大，而且用起来也不灵活，因为已有的平台要支持你这种新玩法需要开发新功能来支持。

为了解决易用性以及灵活性问题，我们通过Spark-2.4.0中的Barrier Scheduling来驱动我们的WLDA，这样在正式进入LDA算法前，可以通过Spark的一些操作提前做相关ETL工作，例如文档分词、词频向量化等前序工作。在我之前的[一篇博客](http://sharkdtu.com/posts/spark-mpi.html)中已阐述Barrier Scheduling的用法，官方目前仅仅是实现了一个简易框架，很多细节还没完善，但是要包装我们的WLDA还是有些局限性，主要表现在以下几点：
1. Spark-Task 与 MPI-Process 之间数据交换通过本地文件完成，即Spark Task先将属于它那个partition的数据写到本地磁盘，再启动MPI-Process，由MPI-Process读取，MPI-Process运行完毕后，将结果写到本地，Spark Task再去本地读取出来。整个交换数据的过程有本地磁盘IO的消耗，另外如果数据量较大，本地磁盘可能有爆盘的风险。
2. Spark Task拉MPI-Process的时候，是由单个Task（一般是partitionId=0的task）通过`mpirun`的方式拉起，这需要机器之间ssh免密支持，然而一般的生产集群是不会开启这个特性的。
3. 默认所有Task都会给对应的MPI-Process喂数据，然而WLDA只有doc进程需要输入，word进程并不需要输入，所以我们需要通过一种通用的方式去选择哪些进程该喂数据，哪些进程不该喂数据。

为解决以上几个局限性，我们在Barrier Scheduling基础上做了一些扩展，但是不污染Spark本身的源码，整体实现流程如下图所示。

<img src="/images/spark-mpi-lda-runtime.png" width="500" height="400" alt="spark-mpi-lda-runtime.png" align=center />

上图流程描述如下：
1. Spark-Task通过[MPICH](https://www.mpich.org/)中的Hydra单独拉起拉起MPI-Process，不需要SSH支持，MPI-Process启动后准备好环境，创建“输入”和“输出”命名管道，“输入”命名管道用于Spark向MPI-Process喂数据，“输出”命名管道用户MPI-Process向Spark传回计算结果。
2. 通过一次Barrier操作，等待所有的MPI-Process进程都起来。
3. Spark Task向“输入”命令管道写数据，MPI-Process读取“输入”命名管道，实现Spark给MPI-Process喂数据。
4. MPI-Process开始迭代训练。
5. 训练结束后，MPI-Process写“输出”命名管道，Spark Task读取“输出”命名管道完成结果的收集。
6. 通过一次Barrier操作，等待所有的MPI-Process进程都结束。

该流程解决了Spark Task与MPI-Process之间数据交换问题，同时不借助SSH解决了MPI-Process拉起的问题，整个流程我们将它封装为一个API，通过Scala的隐式转换方式扩充为RDD的一个算子，对于选择性交换数据问题，我们通过API参数来控制，该算子的原型如下：
```scala
/**
 * Execute MPI Program
 *
 * @param mpiProgram The mpi start command
 * @param sendFunc Send the rdd data to mpi program
 * @param sendBlackList Partition id in sendBlackList will not be send data
 * @param outputPath The result output path
 * @param outputBlackList Partition id in outputBlackList will have not output
 */
def mpi(
    mpiProgram: String,
    sendFunc: Option[(Iterator[T], OutputStream) => Unit] = None,
    sendBlackList: Set[Int] = Set.empty,
    outputPath: Option[String] = None,
    outputBlackList: Set[Int] = Set.empty): Unit
```
> `mpi`算子为一个action操作，其中： `mpiProgram`为mpi二进制启动命令；`sendFunc`为用户自定义Spark给MPI-Process喂数据的逻辑，默认Spark将每条记录以文本形式喂给MPI-Process；`sendBlackList`指示哪些partition不用喂数据，`outputPath`为最终结果保存路径，`outputBlackList`指示哪些partition没有输出。

如果需要通过Spark驱动一个MPI程序，可以通过如下调用方式实现：
```scala
rdd.mpi(...)
```

对于WLDA，借助我们扩充的`mpi`算子很容易按如下方式实现：
```scala
class WLDAPartitioner(override val numPartitions: Int, numWordProcesses: Int) extends Partitioner {
  override def getPartition(key: Any): Int = {
    val pid = key.asInstanceOf[Long] % (numPartitions - numWordProcesses) + numWordProcesses
    pid.toInt
  }
}

object WLDA {
  def main(args: Array[String]): Unit = {
    // val params = ...
    val rdd = sc.textFile(params.inputPath)
      .zipWithIndex()
      .map { x => (x._2, x._1) }
      .reduceByKey(
        new WLDAPartitioner(numPartitions, params.numWordProcesses),
        (a: String, _: String) => a)
      .map(_._2)

    val mpiProgram = s"./wlda " +
      s"--deploy_mode=spark " +
      s"--vocab_size=${params.numVocab} " +
      s"--num_topics=${params.numTopics} " +
      s"--num_iterations=${params.numIterations} " +
      s"--num_word_procs=${params.numWordProcesses}"

    val sendBlackList = (0 until params.numWordProcesses).toSet
    rdd.mpi(mpiProgram, sendBlackList = sendBlackList, outputPath = Some(params.outputPath))
  }
}
```

上述代码显示通过一个自定义的Partitioner，将数据均匀分区到doc processes对应的分区中，word processes对应的分区将不会有数据，`mpi`算子中的`sendBlackList`参数包含word processes对应的分区Id，指示不用给这些分区传数据。具体提交上述Spark程序时，要通过`--files`选项上传二进制文件`mpiexec.hydra`和`wlda`，这些二进制文件编译的时候是以静态链接方式构建，以避免每台机器都装mpi基础库，所以通过这种包装，是可以提交到公司任何Spark平台（例如Tesla、自建Hadoop集群等）运行的。

## 性能

基于上述实现，我们将WLDA与SparkLDA做了一个对比测试，数据集使用[pubmed](https://www.nlm.nih.gov/databases/download/pubmed_medline.html)，820W篇文档，词汇量为141044，训练1000个主题，使用相同的资源(1000 vcore、2T memory)，100轮迭代性能情况如下：
* Spark EM LDA: >350min
* Spark Online LDA: Driver OOM
* WLDA: 12min

可以看到，WLDA性能大幅优于SparkLDA，然而使用上跟SparkLDA无异，对用户无门槛。

## 小结

本文阐述了我们自研的一种LDA实现"WLDA"，它同时基于Spark和MPI，在性能和易用性能方面做到兼顾，该种做法并不是仅对于WLDA有效，可以拓展到其他算法，我们扩展了一个`RDD`的算子`mpi`，该算子可以轻松方便地将Spark与MPI程序协同，可以起到Spark和MPI的桥梁作用。
