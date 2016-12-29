---
title: 深入理解Spark(四)：Spark 底层网络模块
date: 2016-11-12  22:11:29
categories: spark
tags:
  - spark
  - network
  - 大数据
  - 分布式计算
---

对于分布式系统来说，网络是最基本的一环，其设计的好坏直接影响到整个分布式系统的稳定性及可用性。为此，Spark专门独立出基础网络模块spark-network，为上层RPC、Shuffle数据传输、RDD Block同步以及资源文件传输等提供可靠的网络服务。在spark-1.6以前，RPC是单独通过akka实现，数据以及文件传输是通过netty实现，然而akka实质上底层也是采用netty实现，对于一个优雅的工程师来说，不会在系统中同时使用具有重复功能的框架，否则会使得系统越来越重，所以自spark-1.6开始，通过netty封装了一套简洁的类似于akka actor模式的RPC接口，逐步抛弃akka这个大框架。从spark-2.0起，所有的网络功能都是通过netty来实现。<!--more-->

## 系统抽象

在介绍spark网络模块前，我们先温习下netty的基本工作流程。无论是服务器还是客户端都会关联一个channel(socket)，channel上会绑定一个pipeline，pipeline绑定若干个handler，用来专门用来处理和业务有关的东西，handler有UpHandler和DownHandler两种，DownHandler用来处理发包，UpHandler用来处理收包，大致过程如下图所示。

<img src="/images/spark-network-netty-overview.png" width="100%" height="100%" alt="spark-network-netty-overview" align=center />

Spark的底层网络实现也是遵循上图所示流程，其总体实现流程如下图所示。

<img src="/images/spark-network-basic.png" width="100%" height="100%" alt="spark-network-basic" align=center />



### 消息抽象

### Handler定义


## RPC消息处理

## ChunkFetch消息处理

## Stream消息处理
