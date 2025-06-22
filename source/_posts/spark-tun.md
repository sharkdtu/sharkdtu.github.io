---
title: Spark 应用程序调优
date: 2015-06-17  14:22:33
categories: spark
comments: false
tags:
  - spark
  - 大数据
  - 分布式计算
---

对于很多刚接触Spark的人来说，可能主要关心数据处理的逻辑，而对于如何高效运行Spark应用程序了解较少。由于Spark是一种分布式内存计算框架，其性能往往受限于CPU、内存、网络等多方面的因素，对于用户来说，如何在有限的资源下高效地运行Spark应用程序显得尤为重要。下面只针对Spark-On-Yarn的一些常用调优策略做详细分析。<!--more-->

## 配置参数优化

### 资源申请参数

Spark-On-Yarn资源调度由Yarn来管理，用户只需指定Spark应用程序要申请的资源即可。我们首先来理解几个资源配置项，一旦资源配置确定，则只能在这些有限的资源下运行Spark应用程序。

* num-executors：同时运行的executor数。
* executor-cores：一个executor上的core数，表示一次能同时运行的task数。一个Spark应用最多可以同时运行的task数为num-executors*executor-cores。
* driver-memory：driver的内存大小。
* executor-memory：executor内存大小，视任务处理的数据量大小而定。

一开始我们只能通过大致的估算来确定上述资源的配置，例如一个Spark应用程序处理的数据大小为1T，如果读出来默认是500个partitions（可以通过测试运行，从web中查看的到），那么平均每个partition的大小为1T/500≈2G，默认情况下，考虑中间处理过程中的数据膨胀以及一些额外内存消耗，executor中可用于存放rdd的阈值设定为`spar.storage.memoryFraction=0.6`，所以存储partition需要的内存为`executor-memory*0.6`，稳妥一点设置executor-memory大于2G/0.6，如果一个executor不止是处理一个partition，假如num-executors设置为100，那么平均每个executor处理的partition为500/100=5，这时如果需要缓存rdd，那么executor-memory就要设置为大于5*2G/0.6；如果读出来的分区数很少（如100），一个partition很大（1T/100≈10G），使得executor-memory有可能OOM，那么就需要考虑加大分区数（调用repartition(numPartitions)等），增加task数量来减少一个task的数据量。一般来说一个executor处理的partition数最好不要超过5个，否则增加num-executors数，接上面的例子，500个分区，配置num-executors为100，每个executor需要处理5个partition。driver-memory的大小取决于最后的action操作，如果是调用collect，那么driver-memory的大小就取决于结果集rdd的大小，如果是调用count，那么driver-memory的大小只需要满足运行需求就够了，对于需要长时间迭代的Spark应用，driver端需要维护rdd的依赖关系，所以需要设置较大的内存。

上述仅仅是大致估算的资源配置，实际还要根据运行情况不断的调优，以达到资源最大化利用。例如，我们在运行日志中找到如下信息，它表明rdd_0的partition1内存大小为717.5KB，当我们得到这个信息后，就可以再次调整上述参数。

    INFO BlockManagerMasterActor: Added rdd_0_1 in memory on mbk.local:50311 (size: 717.5 KB, free: 332.3 MB)

### 运行时参数

(1) spark.serializer

序列化对于Spark应用的性能来说，影响是非常大的，它涉及到网络传输以及存储，Spark默认是使用`org.apache.spark.serializer.JavaSerializer`，内部使用的是Java的`ObjectOutputStream`框架，这种序列化方式压缩比小，而且速度慢，强烈建议采用kyro序列化方式，它速度快，而且压缩比高，性能是Java序列化的10倍，修改配置`spark.serializer=org.apache.spark.serializer.KryoSerializer`即可，一般来说使用kyro序列化方式，需要在程序里面对用户自定义的可序列化的类进行注册，例如下面代码所示：

```scala
valconf = new SparkConf()
conf.registerKryoClasses(Array(classOf[MyClass1], classOf[MyClass2]))
valsc = new SparkContext(conf)
```

但是如果你不注册，kyro也是可以工作的，只是序列化效率差一点。

(2) spark.rdd.compress

这个参数决定了RDD Cache的过程中，RDD数据是否需要进一步压缩再Cache到内存或磁盘中，从内存看来，当内存比较稀缺时，如果不做压缩就Cache，就很可能会引发GC拖慢程序，从磁盘看来，压缩后数据量变小以减少磁盘IO。所以如果出现内存吃紧或者磁盘IO问题，就需要考虑启用RDD压缩。默认是关闭的。

(3) spark.storage.memoryFraction

前面提到的executor-memory决定了每个executor可用内存的大小，而spark.storage.memoryFraction则决定了在这部分内存中有多少可以用于管理RDD Cache数据，剩下的内存用来保证任务运行时各种其它内存空间的需要。`spark.executor.memoryFraction`默认值为0.6，官方文档建议这个比值不要超过JVM Old Gen区域的比值，因为RDD Cache数据通常都是长期驻留内存的，理论上也就是说最终会被转移到Old Gen区域，如果这部分数据允许的尺寸太大，势必把Old Gen区域占满，造成频繁的FULL GC。如果发现Spark应用在运行过程中发生频繁的FULL GC，就需要考虑减小该配置，所以建议这个配置不要加大，如果内存吃紧，可以考虑采用内存和磁盘的混合缓存模式，进一步减少RDD Cache还可以考虑序列化以及压缩等。

(4) spark.shuffle.memoryFraction

在启用Spill的情况（`spark.shuffle.spill`默认开启）下，`spark.shuffle.memoryFraction`表示Shuffle过程中使用的内存达到总内存多少比例的时候开始Spill。`spark.shuffle.memoryFraction`默认值为0.2，调整该值可以调整Shuffle过程中Spill的频率。总的来说，如果Spill太过频繁，可以适当增加`spark.shuffle.memoryFraction`的大小，增加用于Shuffle的内存，减少Spill的次数。然而这样一来为了避免内存溢出，对应的可能需要减少RDD cache占用的内存，即减小`spark.storage.memoryFraction`的值，这样RDD cache的容量减少，有可能带来性能影响，因此需要综合考虑，如果在你的Spark应用程序中RDD Cache较少，Shuffle数据量较大，就需要把`spark.shuffle.memoryFraction`调大一些，把`spark.storage.memoryFraction`调小一些。

(5) spark.shuffle.file.buffer.kb

每次shuffle过程驻留在内存的buffer大小，在shuffle中间数据的产生过程中可减少硬盘的IO操作。`spark.shuffle.file.buffer.kb`默认为32，若Spark应用程序运行过程中Shuffle称为瓶颈，根据需要适当的加大该配置。

## 接口使用优化

对于Spark新手来说，可能不太了解RDD接口内部实现细节，主要关心业务数据处理，然而这往往导致编写出来的Spark应用程序运行效率不高，资源利用浪费等。下面简单介绍一些常见的Spark应用开发注意细节。

### 缓存接口

Spark比MapReduce快的很大一部分原因是它可以把中间结果RDDCache起来，不用每次需要时重新计算。但是如果Cache使用不当，会造成内存吃紧，要么带来不必要的磁盘IO，要么引起频繁的FULL GC，拖慢程序运行。

对于一个需要多次使用的临时RDD（类似于临时变量），尽可能要把它Cache起来，这样这个临时RDD只会计算一次，以后每次都会从Cache里直接取。如下面的例子，需要统计第一个字段大于100的数目和第二个字段大于100的数目，如果data不做Cache，因为只有遇到RDD的Action接口时才出发计算，所以在计算firstCnt时会读一遍数据，计算secondCnt时还会再读一遍数据，这样就造成一些不必要的计算，对data做了Cache后，在计算firstCnt时读一次，计算secondCnt就会直接从Cache中取而不用再次读一次。

```scala
val data = sc.textFile(path)
data.cache()
val firstCnt = data.filter(x(0).toInt => 100).count()
val secondCnt = data.filter(x(1).toInt => 100).count()
```

很多时候会看到这样的代码，在对两个RDD进行Join时，把两个RDD都Cache起来再做Join，这里一定要明白一点，没有调用Action接口，计算是不会触发的，下面的代码如果后续不再用到rdd1和rdd2，是没有必要对rdd1和rdd2做Cache的，这里要做Cache的是data。

```scala
val data = val data = sc.textFile(path)
val rdd1 = data.map(…).cache()
val rdd2 = data.map(…).cache()
val rdd3 = rdd1.join(rdd2).count()
```

对于内部需要多次迭代的Spark应用来说，应该尽量将每次迭代用到的临时RDD缓存起来，在这个临时RDD被更新时，需要将旧的缓存手动清除掉。如下例子显示，每次迭代都需要在curRDD基础上进行更新得到updatedRDD，在一轮迭代结束后要更新curRDD为updatedRDD，在更新前手动将之前的curRDDCache清理掉，防止内存被耗光，引发频繁FULL GC。

```scala
val data = sc.textFile(path)
// some transformations in init(data)
varcurRDD = init(data).cache()
val result = new ArrayBuffer[Double]()
// some transformations and an action in getResult(curRDD)
result += getResult(curRDD)
// Start Iteration
var changed = true
while(changed) {
  // some transformations in iteration(curRDD)
  valupdatedRDD = iteration(curRDD).cache()
  // getResultand check if the value is changed
  val x = getResult(updatedRDD)
  // convergence
  if(x == result.last) changed = false
  // Unpersist old RDD and assign new RDD
  curRDD.unpersist(false)
  curRDD = updatedRDD
}
```

在对RDD做缓存时，还应考虑内存大小情况选择合适的缓存方式，Spark提供以下几种缓存：

* MEMORY_ONLY：直接将RDD对象保存到内存中，Spark默认选项
* MEMORY_AND_DISK：当内存不够的时候，保存到磁盘中（内存较为稀缺的时候用，比MEMORY_ONLY占用更少的内存，但是会带来磁盘IO）
* MEMORY_ONLY_SER：将RDD序列化后保存到内存中（内存较为稀缺的时候用，比MEMORY_ONLY占用更少的内存）
* MEMORY_AND_DISK_SER：将RDD序列化后保存到内存中，内存不够时保存到磁盘中（内存较为稀缺的时候用，比MEMORY_ONLY_SER更安全）
* DISK_ONLY：保存到磁盘中（不建议用）
* MEMORY_ONLY_2：与MEMORY_ONLY类似，只是保存两份
* MEMORY_AND_DISK_2：与MEMORY_AND_DISK类似，只是保存两份
* OFF_HEAP ：将序列化后的RDD保存到Tachyon（一种分布式内存文件系统）中，相比于MEMORY_ONLY_SER可以避免GC的额外开销。这种缓存方式还在试验阶段

根据具体情况判断使用何种缓存方式，调用的时候直接通过如`rdd.persist(StorageLevel.MEMORY_AND_DISK_SER)`方式实现，调用`rdd.cache()`默认是`rdd.persist(StorageLevel.MEMORY_ONLY)`。

### 引发Shuffle的相关接口
一个Spark应用程序运行快慢，往往受限于中间的Shuffle过程，Shuffle涉及到网络以及磁盘IO，是整个Spark应用程序运行过程中较为耗时的阶段。在编写Spark应用程序时，应当尽量减少Shuffle次数。下面列举常见的可能引发Shuffle的接口。

* distinct
* Intersection/subtracted
* reduceByKey/aggregateByKey
* repartition
* cogroup
* join
* sortBy/sortByKey
* groupBy/groupByKey
* partitionBy

如果executor内存不足以处理一个partition，那么这时考虑调用repartition来加大分区数，使得每个partition的数据量减少以至于executor可以处理，一般来说上述接口也可以接受numPartitions参数来指定分区数。上述接口连续调用不一定会带来多次Shuffle，只要partition类型和partition数不变，是不会增加Shuffle次数的，如下代码则只有一次Shuffle：

```scala
rdd.map(x => (x, x+1)).repartition(1000).reduceByKey(_ + _).count()
```

然而如下代码却会有两次Shuffle：

```scala
rdd.map(x => (x, x+1)).repartition(1000).reduceByKey(_ + _, 3000).count()
```

很多人在一开始调用了触发Shuffle的相关接口，后面可能数据膨胀了，发现需要更多的partition，所以在后面调用触发Shuffle的相关接口时加大partition数，这样就会导致多次Shuffle，所以一开始就确定好最后的partition数，以免做不必要的Shuffle。

### 接口对比

(1) sortBy/sortByKey与takeOrdered

有时候我们可能希望对数据集排序取前n条记录，很多人会像如下代码一样实现：

```scala
rdd.sortBy(x => x.key).take(n)
//or rdd.sortByKey().take(n)
```

然而，有一个更有效的办法，就是按照以下方式实现：

```scala
rdd.takeOrdered(n)
```

以上两者的区别在于，第一种方式需要把所有partition的排序结果进行归并再取前n条记录，第二种方式是从每个排好序的partition中取出前n条记录最后再归并为n条记录，大大降低了网络IO，提升整体性能。

(2) groupBy/groupByKey与aggregateByKey

在做分组计算时，首先会想到使用`groupBy/groupByKey`接口，值得一提的是，`groupBy/groupByKey`接口特别占用内存，它是把具有相同key值的所有value放到一个buffer数组里，如果某个key对应的value非常多，极其容易引发OutOfMemoryError，通过`groupBy/groupByKey`实现的分组计算功能是可以通过`aggregateByKey`或者`reduceByKey`来实现的，`aggregateByKey/reduceByKey`内部是通过combineByKey实现的，当内存超过一定阈值会spill到磁盘，相对来说较为安全。当通过`groupBy/groupByKey`接口最后返回的`RDD[(K, V)]`中`V`不是序列时，可以用`reduceByKey`实现，当`V`是序列时可以用`aggregateByKey`实现，例如需要统计key对应的value最大值：

```scala
//rdd: RDD[(int, int)]
rdd.groupByKey().map((k, vb) => (k, vb.max))
```

我们完全可以用reduceByKey来实现上述功能：

```scala
rdd.reduceByKey ((v1, v2) => Math.max(v1, v2))
```

再比如，就想返回key对应的所有value：

```scala
//rdd: RDD[(int, int)]
rdd.groupByKey()
```

我们完全可以用aggregateByKey来实现上述功能：

```scala
rdd. aggregateByKey(Seq ())(
(u, v) => v::u,
(u1, u2) => u1 ++ u2
)
```

以上是简单提出几个需要注意的接口调用，如果不了解RDD接口的使用，可以参见[社区文档](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.package)。
