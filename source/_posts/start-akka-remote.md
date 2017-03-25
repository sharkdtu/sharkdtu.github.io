---
title: Akka-remote使用入门
date: 2014-12-20  20:15:33
categories: 程序语言
comments: false
tags:
  - scala
  - akka
  - actor-remote
---

在上一篇文章中讲了[akka-actor的简单使用](/posts/start-akka-actor.html)，那主要是展现了akka在一台机器上的并发应用，这一篇接着介绍akka-remote使用，简单了解akka在不同机器上的并发应用。我们知道，在一台机器上是多个本地actor之间发送消息，那么如果是在多台机器上，则不同机器上的actor是通过网络通信来发送消息的。假如，我们还是用之前的蒙特卡洛求π的例子，之前在一台机器上计算启动多个actor同时计算，其根源还是利用的一台机器的资源，如果将计算任务分发到多台机器上同时运行，最后汇总下多台机器的计算结果，每台机器的计算任务就不会太大。<!--more-->
如下图所示，每个方框代表一台机器，我们通过 driver actor 发起计算到 master actor ， master在本地创建一个路由 actor workerRouter来分发任务，任务需要在不同的机器上运行，最后汇总不同机器的计算结果，得到π的近似值。

![akka-remote](/images/akka-remote.png)

同样首先创建sbt项目，在项目根目录下新建sbt项目描述文件`build.sbt`：

    name := "akka-remote-pi"
    [空行]
    version := "1.0.0"
    [空行]
    scalaVersion := "2.10.4"
    [空行]
    libraryDependencies += "com.typesafe.akka" %% "akka-actor" % "2.3.7"
    [空行]
    libraryDependencies += "com.typesafe.akka" %% "akka-remote" % "2.3.7"

我们定义actor之间的消息格式，与上一篇文章中[akka-actor](/posts/start-akka-actor.html)消息一样：

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

worker actor也与上一篇一样，收到分发的任务后，通过随机掷点的方法统计落在内切圆中点个数：

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

worker actor需要启动远程通信功能，需要在项目资源目录里添加`worker.conf`配置：

    akka {
      actor {
        provider = "akka.remote.RemoteActorRefProvider"
      }
      remote {
        netty.tcp {
          //这里是worker actor远程通信地址和端口，
          //我这里是在本机上测试的
          hostname = "127.0.0.1"
          port = 2554
        }
      }
    }

下面定义 master actor，与上一篇有点不一样，这里利用了配置文件功能，在创建`WorkerRouter`时从配置中加载路由配置，`nrOfMessages`和`times`也从配置文件中读取，这就不需要直接放到启动参数中去了。

```scala
class Master extends Actor {
    // 计算发起者
    var calculateSender: ActorRef = _
    // worker计算结果的累加统计
    var acc: Int = 0
    // 记录收到的结果消息个数
    var nrOfResults: Int = 0
    // 计算启动时间
    val start: Long = System.currentTimeMillis

    // 路由actor
    val workerRouter = context.actorOf(
      FromConfig.props(Props[Worker]),
      name = "workerRouter")

    val cfg = context.system.settings.config
    val nrOfMessages: Int = cfg.getInt("parameters.nrOfMessages")
    val times: Int = cfg.getInt("parameters.times")

    def receive = {
      case Calculate =>
        calculateSender = sender // 先保存计算发起者的ActorRef，以便后续回复结果
        for (i <- 0 until nrOfMessages) workerRouter ! Work(times)

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
```

master actor需要启动远程通信功能，需要在项目资源目录里添加`master.conf`配置文件：

    akka {
      actor {
        provider = "akka.remote.RemoteActorRefProvider"
        deployment {
          "/master/workerRouter" {
            router = round-robin-pool
            nr-of-instances = 10 //在target.nodes机器上启动10个worker
            target.nodes = [ //路由分发任务的worker actor地址
            //  "akka.tcp://WorkerSystem@127.0.0.1:2554",
              "akka.tcp://WorkerSystem@127.0.0.1:2554"
            ]
          }
        }
      }
      remote {
        netty.tcp {
          //这里是master actor远程通信地址和端口，
          //我这里是在本机上测试的
          hostname = "127.0.0.1"
          port = 2553
        }
      }
    }
    parameters {//参数
      nrOfMessages = 1000
      times = 1000
    }

下面我们开始定义 driver actor，它需要一个master地址参数，主要是通过`actorSelection`函数查询到 master actor，并向其发起计算，最后得到计算结果：

```scala
class Driver(masterPath) extends Actor {
    def receive = {
      case Calculate =>
        context.actorSelection(masterPath) ! Calculate

      case PiApproximation(pi, duration) =>
        println(
          "Pi: \t" + pi + "\n" +
          "Spend: \t" + duration)
        context.system.shutdown()
    }
}
```

类似地，需要开始远程功能，则添加配置`driver.conf`：

    akka {
      actor {
        provider = "akka.remote.RemoteActorRefProvider"
      }
      remote {
        netty.tcp {
          //这里是worker actor远程通信地址和端口，
          //我这里是在本机上测试的
          hostname = "127.0.0.1"
          port = 0
        }
      }
    }

所有actor和配置都定义完了，下面我们开始分别编写启动actor的函数以及入口函数：

```scala
def startWorkerSystem(): Unit = {
  ActorSystem("WorkerSystem", ConfigFactory.load("worker"))
  println("Started WorkerSystem")
}

def startMasterSystem(): Unit = {
  val system = ActorSystem("MasterSystem", ConfigFactory.load("master"))
  system.actorOf(Props(classOf[Master]))
  println("Started MasterSystem")
}

def startDriverSystem(): Unit = {
  val system = ActorSystem("DriverSystem", ConfigFactory.load("driver"))
  val masterPath = "akka.tcp://MasterSystem@127.0.0.1:2553/user/master"
  val driver = system.actorOf(Props(classOf[Driver], masterPath),
    name = "driver")
  driver ! Calculate
}

def main(args: Array[String]) {
  if (args.isEmpty && args.head == "worker")
    startWorkerSystem()
  if (args.isEmpty && args.head == "master")
    startMasterSystem()
  if (args.isEmpty && args.head == "driver")
    startDriverSystem()
}
```

用sbt编译后，可以在一个终端里直接`sbt run`运行，也可以开三个终端，先开启worker，然后开启master，最后开启driver。
