---
title: Flink RPC 详解
date: 2021-07-20  20:10:45
categories: flink
comments: false
tags:
  - flink
  - rpc
---

要理解 Flink 内部各组件交互的源码实现，首先必须要理解其 RPC 的工作机制。与 Hadoop、Spark 等系统类似，作为一个独立的分布式系统框架，Flink 也抽象了自己的一套 RPC 框架，本文尝试尽可能详尽地阐述其设计及实现原理。<!--more-->

## 接口设计

首先不用纠结其内部实现细节，先感性地认识下如何使用 Flink RPC 框架实现一个基本的 RPC 调用。

1. 定义接口协议
```java
public interface HelloGateway extends RpcGateway {
    CompletableFuture<String> sayHello();
}
```

2. 服务端组件实现接口
```java
// RpcEndpoint 可以理解为服务端组件
public static class HelloEndpoint extends RpcEndpoint implements HelloGateway {
    protected HelloEndpoint(RpcService rpcService) {
        super(rpcService);
        ...
    }
    @Override
    public CompletableFuture<String> sayHello() {
        return CompletableFuture.completedFuture("Hello World");
    }
}
```

3. 实例化服务端组件
```java
// RpcService 可以理解为 RPC 框架引擎（客户端和服务端都有），可以用来启动、停止、连接一个服务端组件
RpcService rpcService = getRpcService ...
HelloEndpoint helloEndpoint = new HelloEndpoint(rpcService);  // 内部会启动这个组件服务
```

4. 客户端发起远程调用
```java
RpcService rpcService = getRpcService ...
// rpcAddress 唯一标识要连接的服务端组件，例如 "rpc://host:port/path/to/helloendpoint"
HelloGateway helloGateway = rpcService.connect(rpcAddress, HelloGateway.class);
// 如果客户端跟服务端组件在同一个进程里，可以省去connect
// HelloGateway helloGateway = helloEndpoint.getSelfGateway(HelloGateway.class);
helloGateway.sayHello();  // helloGateway 作为客户端代理调用远程方法
```

从以上四步可以看到，Flink RPC 的封装比较高层，客户端的远程调用看起来完全就是调用本地方法，毫无收发消息的痕迹，接口类的命名也比较形象，如下图所示，当要发起远程调用时，临时拿到对应的接口网关，直接调用对应的接口。

<img src="/images/flink-rpc-abstract.png" width="600" height="400" align=center />

有了一个基本的高层次认识后，再仔细分析上述代码，提出几个问题：
1. 服务端组件（RpcEndpoint）实例化过程中做了什么？
2. 我们只是定了接口协议，接口网关（RpcGateway）是如何实例化出来的？
3. 通过接口网关（RpcGateway）调用方法时，其内部是怎么收发消息的？

在具体回答以上三个问题前，先简单介绍下 Java 的动态代理技术。

### Java 动态代理简介

有一种设计模式叫代理模式，通过代理对象访问目标对象，可以在不修改原目标对象的前提下，提供额外的功能操作，以达到扩展目标对象的功能。其UML大致如下图所示。

<img src="/images/flink-rpc-proxy-pattern.png" width="600" height="400" align=center />

代理模式在 Java 中有静态代理和动态代理之分，我们先看下静态代理：
```java
public interface HelloInterface {
    String sayHello();
}

public class ChinaHello implements HelloInterface {
    @Override
    public String sayHello() {
        return "你好";
    }
}

public class HelloProxy implements HelloInterface {
    private HelloInterface target;

    public HelloProxy(HelloInterface target) {
        this.target = target;
    }

    @Override
    public String sayHello() {
        // do something before
        return target.sayHello();
    }
}
```

以上静态代理模式的代码相信大家或多或少都有见过，通过`HelloProxy`去代理实际目标对象，扩展相关功能。但是静态代理需要在编译时实现，冗余代码较多。另外，Java 也提供了动态代理模式的实现，不需要事先实现接口，运行时通过反射动态实例化特定接口的实例，上述静态代理模式代码可以用如下动态代理模式来实现。
```java
public interface HelloInterface {
    String sayHello();
}

HelloInterface helloProxy = (HelloInterface) Proxy.newProxyInstance(
        getClass().getClassLoader(),
        new Class<?>[] {HelloInterface.class},
        new InvocationHandler() {
            @Override
            public Object invoke(Object proxy, Method method, Object[] args) {
                if (method.getName().equals("sayHello")) {
                    return "你好";
                }
                return null;
            }
        });
helloProxy.sayHello();
```

可以看到以上代码并没有显示地实现接口`HelloInterface`，但是通过 Java 提供的`Proxy.newProxyInstance`方法可以动态创建该接口的实例，当调用该实例的方法时，会被转发到`InvocationHandler#invoke`中。认识了动态代理后，下面回过头来逐一回答前面提到的三个问题。

### 接口实现规范

为了阅读方便，下面先把前面的三个问题再拎出来：
1. 服务端组件（RpcEndpoint）实例化过程中做了什么？
```java
HelloEndpoint helloEndpoint = new HelloEndpoint(rpcService);
```
2. 我们只是定了接口协议，接口网关（RpcGateway）是如何实例化出来的？
```java
HelloGateway helloGateway = rpcService.connect(rpcAddress, HelloGateway.class);
// or
HelloGateway helloGateway = helloEndpoint.getSelfGateway(HelloGateway.class);
```
3. 通过接口网关（RpcGateway）调用方法时，其内部是怎么收发消息的？
```java
helloGateway.sayHello();
```

#### 服务端组件（RpcEndpoint）初始化

为了更简单地处理多线程并发问题，对同一个`RpcEndpoint`的所有调用被设计成在同一个主线里串行执行，所以每个`RpcEndpoint`在实现的时候都不用担心数据共享一致性问题（不用考虑加锁等）。从前面的例子可以知道服务端组件实现了接口协议，如果客户端跟服务端在同一个进程中，客户端直接通过`RpcEndpoint#getSelfGateway`拿到`RpcEndpoint`实例调用对应的方法，那么就无法保证对同一个`RpcEndpoint`的所有调用在同一个主线程中串行执行。

为此，服务端在实例化具体`RpcEndpoint`时，其内部会启动一个`RpcServer`（不对外暴露），`RpcServer`只是一个接口，要实例化一个特定的`RpcServer`实例，就需要通过前面介绍的动态代理技术，在运行时动态生成，UML关系如下图所示。

<img src="/images/flink-rpc-endpoint.png" width="600" height="400" align=center />

通过动态代理生成的`RpcServer`实例会绑定其对应的`RpcEndpoint`所实现的接口协议，即上述例子中`HelloEndpoint`中的`RpcServer`会有`sayHello`方法，所以当客户端跟服务端在同一个进程中，客户端通过`RpcEndpoint#getSelfGateway`拿到其中的`RpcServer`实例作为接口网关，进而调用其绑定的接口协议方法，根据Java动态代理原理，对`RpcServer`中的方法调用会被转发给`InvocationHandler`，在`InvocationHandler`中控制所有调用在同一个主线里串行执行。

#### 客户端获取接口网关（RpcGateway）

客户端发起RPC调用前，需要先拿到对应的接口网关`RpcGateway`，前面介绍到，当客户端与服务端在同一个进程中，通过`RpcEndpoint#getSelfGateway`获取，实际是拿到的是`RpcEndpoint`中的`RpcServer`实例，因为它是通过动态代理绑定了特定的RpcGateway创建的，所以也可以作为`RpcGateway`。当客户端与服务端不在一个进程中，通过`RpcService#connect`获取，服务端的每个`RpcEndpoint`都有一个唯一的 RPC 地址，客户端通过这个地址去连接路由到指定的`RpcEndpoint`，拿到消息 handler，通过消息 handler 双方握手成功后，客户端再通过动态代理创建特定的`RpcGateway`实例，其总体流程如下图所示。

<img src="/images/flink-rpc-new-gateway.png" width="600" height="400" align=center />


#### 客户端发起 RPC 调用

无论客户端是否与服务端在同一个进程中，客户端与`RpcGateway`的UML关系抽象如下图所示。

<img src="/images/flink-rpc-gateway-call.png" width="600" height="400" align=center />

当客户端通过`RpcGateway`调用方法时，根据动态代理原理，该调用会被转发到`InvocationHandler`中，`InvocationHandler`将方法名、参数类型、参数对象列表打包成`RpcInvocation`消息，通过其握有的消息 handler，发送消息，并接受服务端响应，完成一次RPC调用。

## 基于 Akka 的实现

以上只是在一个抽象的层面介绍了 Flink RPC 的设计，具体实现还需要借助一套消息系统来完成，目前 Flink RPC 的默认是基于 Akka 框架实现（也是唯一的实现），Akka 的核心是 Actor 模型，如下图所示，Actor 与 Actor之前只能用消息进行通信，每个Actor都有对应一个信箱，消息是有顺序地被投递到信箱，Actor 串行处理信箱中的消息。建议自行先了解 Akka 及 Actor 的相关知识，这里不展开详细介绍。

<img src="/images/flink-rpc-actor.png" width="600" height="400" align=center />

基于 Actor 模型，每个 RpcEndpoint 关联一个 Actor，正好契合了对每个 RpcEndpoint 的调用要求在同一个线程中完成的设计，同时，Akka 的每个 Actor 都有一个唯一的地址，正好作为 RpcEndpoint 的 RPC Address。

具体实现上，`AkkaRpcService`实现启动、停止、连接一个服务端组件，`AkkaRpcService`内部持有一个`ActorSystem`实例，当启动一个服务端组件时，会创建一个 `AkkaRpcActor`（其中定义了消息处理逻辑，当收到`RpcInvocation`消息时，会按照方法名调用`RpcEndpoint`中的具体实现），作为前面提到的消息 handler，然后通过动态代理实例化一个 `RpcServer`，绑定一个`AkkaInvocationHandler`，其持有前面创建 `ActorRef`。

当客户端与服务端在同一个进程中，那么直接获取这个`RpcServer`实例作为接口网关`RpcGateway`，这样接口网关`RpcGateway`上的方法调用会被转到`AkkaInvocationHandler`中，进而将方法名、参数类型、参数对象列表打包成`RpcInvocation`消息通过 `ActorRef` 发送，其实现如下图UML所示。

<img src="/images/flink-rpc-akka.png" width="600" height="400" align=center />

当客户端与服务端不在同一个进程中，其通过`AkkaRpcService#connect`方法，连接服务端对应的`AkkaRpcActor`以得到其对应的`ActorRef`，类似地，通过动态代理实例化特定的 `RpcGateway`，绑定一个`AkkaInvocationHandler`，其持有前面连接获取到的 `ActorRef`，之后这个接口网关`RpcGateway`的方法调用会被转到`AkkaInvocationHandler`中，进而将方法名、参数类型、参数对象列表打包成`RpcInvocation`消息通过 `ActorRef` 发送，其实现如下图UML所示。。

<img src="/images/flink-rpc-akka2.png" width="600" height="400" align=center />

## 小结

某种程度上讲，Flink RPC 设计来源于 Actor 模型，只是在这之上做了更高层的抽象，应用层不感知底层的消息收发，做到如同本地方法调用一般。一开始看其源码实现可能会觉得过度设计，来回绕圈，但是当看懂其设计本意后，就会觉得别有一番风味，里面包含了大量的优秀设计模式，对于我们实际写代码有很大的参考价值。
