---
title: Akka-actor使用入门
date: 2014-12-12  21:45:11
categories: 程序语言
comments: false
tags:
  - scala
  - akka
  - actor
---

学习scala编程，不可避免的会接触到actor模式，它使得并发编程不再像噩梦般萦绕着开发者，Akka是actor的一个开源实现。由于本人水平有限，自认为还不能把actor设计思想讲明白，所以本文仅仅是一个使用akka-actor的入门参考以及个人的入门心得，其具体原理及设计思想请参考相关资料，推荐[Akka的官方文档](http://doc.akka.io/docs/akka/2.3.7/general/actor-systems.html)，里面讲的很清晰，另外，[国外一个人的笔记](http://rerun.me/2014/09/11/introducing-actors-akka-notes-part-1/)写的相当不错，原理加上其配图讲的非常形象。<!--more-->

下面我通过一个简单例子来阐述akka-actor的使用，假如我们需要编写一个程序，利用[蒙特卡洛算法](https://zh.wikipedia.org/wiki/%E8%92%99%E5%9C%B0%E5%8D%A1%E7%BE%85%E6%96%B9%E6%B3%95)求圆周率π，我们知道蒙特卡洛算法是采用统计模拟方法，在一个边长为2的正方形内，一个点落在其内切圆内的概率为π/4（圆的面积/正方形的面积），编写程序时，假设有多个actor同时随机在正方形内掷点，统计所有actor掷的点，把那些落在内切圆内的点的个数加起来再除以所有actor掷的总次数，即为一个点落在内切圆内概率的近似值，进而可以得到π的近似值。

首先我们通过sbt创建项目，根据sbt项目的目录结构生成项目的雏形（可以将以下命令写到一个脚本里，以后重复使用）：

    [hadoop@master ~]$ mkdir -p akka-actor-pi/lib
    [hadoop@master ~]$ mkdir -p akka-actor-pi/project
    [hadoop@master ~]$ mkdir -p akka-actor-pi/src/main/scala
    [hadoop@master ~]$ mkdir -p akka-actor-pi/src/main/resources
    [hadoop@master ~]$ mkdir -p akka-actor-pi/src/test/scala

在项目根目录akka-actor-pi下新建sbt项目描述文件`build.sbt`，注意每行之间有一个空行:

    name := "akka-actor-pi"
    [空行]
    version := "1.0.0"
    [空行]
    scalaVersion := "2.10.4"
    [空行]
    libraryDependencies += "com.typesafe.akka" %% "akka-actor" % "2.3.7"

在project目录里添加`build.properties`文件，在其中声明sbt的版本信息：

    sbt.version=0.13.6

如下图所示为程序求π的actor系统结构图，程序入口通过向master发起计算请求，master将任务分发到许多worker上运行（类似于启动多个线程并行计算），中间通过一个路由actor来间接的分发任务，把路由actor看成网络中的路由器，当其收到任务请求后，它会按某种调度方式来分发到它关联的那些actor上，在我们的程序实现中，采用轮询的方式挨个分发任务。

![akka-actor](/images/akka-actor.png)

下面开始定义这个系统中流动的消息，我们需要四种消息：

* Calculate： 发送给master来启动计算；
* Work： 从master发送给各worker，包含工作分配的内容；
* Result： 从worker发送给master，包含worker的计算结果；
* PiApproximation： 从master返回给调用端，包含π的最终计算结果和整个计算耗费的时间；

发送给actor的消息应该永远是不可变的，在scala里有`case classes`来构造完美的消息，下面是用`case class`创建四种消息，其中创建一个通用的基础`trait`(定义为`sealed`以防止在其他地方创建消息)。

```scala
sealed trait PiMessage
// 计算启动消息
case object Calculate extends PiMessage
// 分配的任务，需要做几次投掷测试
case class Work(times: Int) extends PiMessage
// Worker一次任务计算结果，包含落在内切圆里的点的个数
case class Result(value: Int) extends PiMessage
// 总的计算结果，包含π的近似值与计算所花的时间
case class PiApproximation(pi: Double, duration: Duration)
    extends PiMessage
```

下面定义 worker actor，收到分发的任务后，通过随机掷点的方法统计落在内切圆中点个数。

```scala
class Worker extends Actor {
  // 蒙特卡洛测试
  def testMonteCarlo(times: Int): Int = {
    var acc = 0
    for (i <- 0 until times) {
      val x = random * 2 - 1 //生成[-1,1]区间的横坐标
      val y = random * 2 - 1 //生成[-1,1]区间的纵坐标
      if (x * x + y * y < 1) acc += 1
    }
    acc
  }

  def receive = {
    case Work(times) =>
      sender ! Result(testMonteCarlo(times))
  }
}
```

下面定义 master actor，主要是给一些worker分发计算任务，收集Worker的计算结果，然后计算出π的近似值，master包含三个参数，`nrOfWorkers`表示一共有几个Worker同时运行，`nrOfMessages`表示要发多少任务下去（一个worker有可能会运行多个任务），`times`表示Worker运行一次任务需要掷几次点。

```scala
class Master(nrOfWorkers: Int, nrOfMessages: Int, times: Int)
  extends Actor {
  var calculateSender: ActorRef = _  //保存计算发起者的ActorRef
  var acc: Int = 0
  var nrOfResults: Int = 0
  val start: Long = System.currentTimeMillis

  // 创建一个路由actor来分发任务
  val workerRouter = context.actorOf(
    Props[Worker].withRouter(RoundRobinRouter(nrOfWorkers)),
    name = "workerRouter")

  def receive = {
    case Calculate =>
      for (i <- 0 until nrOfMessages) workerRouter ! Work(times)
      calculateSender = sender

    case Result(value) =>
      acc += value
      nrOfResults += 1
      if (nrOfResults == nrOfMessages) {
        // 所有Worker工作完毕，计算π的近似值，并给发起者回复
        val pi = (4.0 * acc) / (nrOfMessages * times)
        calculateSender ! PiApproximation(pi,
          (System.currentTimeMillis - start).millis)
      }   
  }
}
```

定义完所有actor的功能后，我们可以开始编写入口程序，来使用这些actor完成π的近似计算了，程序首先创建 master actor的引用，发送计算请求消息，等待计算结果，注意这里是通过ask的方式（？标识符）发送消息，前面我们是通过tell（！标识符）方式发送消息，这两种方式的区别在于，tell方式发送消息时发出后不管的，一般在actor之间用这种方式发消息，reply消息可以通过tell方式发回来，而ask方式发送消息是期望得到一个结果的，其返回类型是`Future`类型，最后做一个转换，得到所期望的消息，如下代码是通过`Await`阻塞等待结果。

```scala
object Pi {
  implicit val timeout = Timeout(100 seconds)

  def main(args: Array[String]) {
    if (args.length < 3) {
      System.err.println("Usage: Pi <nrOfWorkers> <nrOfMessages> <times>")
      System.exit(1)
    }
    val system = ActorSystem("PiSystem")
    val master = system.actorOf(Props(new Master(
      args(0).toInt, args(1).toInt, args(2).toInt)),
      name = "master")
    val future = master ? Calculate
    val approximationPi = Await.result(future, timeout.duration)
                               .asInstanceOf[PiApproximation]
    println("Pi: \t" + approximationPi.pi)
    println("Spend: \t" + approximationPi.duration)
    system.shutdown()
  }
}
```

我们再把上述代码重新整理一遍，在项目代码目录`src/main/scala`中新建文件`Pi.scala`。

```scala
import scala.math.random
import scala.concurrent.Await
import scala.concurrent.duration._

import akka.actor._
import akka.routing.RoundRobinRouter
import akka.pattern.ask
import akka.util.Timeout


object Pi {

  sealed trait PiMessage

  case object Calculate extends PiMessage

  case class Work(times: Int) extends PiMessage

  case class Result(value: Int) extends PiMessage

  case class PiApproximation(pi: Double, duration: Duration)
    extends PiMessage

  class Worker extends Actor {
    def testMonteCarlo(times: Int): Int = {
      var acc = 0
      for (i <- 0 until times) {
        val x = random * 2 - 1
        val y = random * 2 - 1
        if (x * x + y * y < 1) acc += 1
      }
      acc
    }

    def receive = {
      case Work(times) =>
        sender ! Result(testMonteCarlo(times))
    }
  }

  class Master(nrOfWorkers: Int, nrOfMessages: Int, times: Int)
    extends Actor {
    var calculateSender: ActorRef = _
    var acc: Int = 0
    var nrOfResults: Int = 0
    val start: Long = System.currentTimeMillis

    val workerRouter = context.actorOf(
      Props[Worker].withRouter(RoundRobinRouter(nrOfWorkers)),
      name = "workerRouter")

    def receive = {
      case Calculate =>
        for (i <- 0 until nrOfMessages) workerRouter ! Work(times)
        calculateSender = sender

      case Result(value) =>
        acc += value
        nrOfResults += 1
        if (nrOfResults == nrOfMessages) {
          val pi = (4.0 * acc) / (nrOfMessages * times)
          calculateSender ! PiApproximation(pi,
            (System.currentTimeMillis - start).millis)
        }

    }
  }

  implicit val timeout = Timeout(100 seconds)

  def main(args: Array[String]) {
    if (args.length < 3) {
      System.err.println("Usage: Pi <nrOfWorkers> <nrOfMessages> <times>")
      System.exit(1)
    }
    val system = ActorSystem("PiSystem")
    val master = system.actorOf(Props(new Master(
      args(0).toInt, args(1).toInt, args(2).toInt)),
      name = "master")
    val future = master ? Calculate
    val approximationPi = Await.result(future, timeout.duration)
      .asInstanceOf[PiApproximation]
    println("Pi: \t" + approximationPi.pi)
    println("Spend: \t" + approximationPi.duration)
    system.shutdown()
  }
}
```

用sbt编译运行项目，并可以看到打印的计算结果。

    [hadoop@master akka-actor-pi]$ sbt compile
    [hadoop@master akka-actor-pi]$ sbt
    [info] Loading project definition from /home/hadoop/project/akka-actor-pi/project
    [info] Set current project to akka-actor-pi (in build file:/home/hadoop/project/akka-actor-pi/)
    > run 4 10000 10000
    [info] Running Pi 4 10000 10000
    Pi:     3.1415648
    Spend:  9678 milliseconds
    [success] Total time: 10 s, completed Dec 12, 2014 10:22:06 PM
