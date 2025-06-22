---
title: CentOS下Hadoop-2.2.0集群安装配置
date: 2014-10-25 19:54:25
categories: hadoop
comments: false
tags:
  - hadoop
  - 大数据
  - 分布式计算
---

对于一个刚开始学习Spark的人来说，当然首先需要把环境搭建好，再跑几个例子，目前比较流行的部署是Spark On Yarn，作为新手，我觉得有必要走一遍Hadoop的集群安装配置，而不仅仅停留在本地(local)模式下学习，因为集群模式下跨多台机器，环境相对来说更复杂，许多在本地(local)模式下遇不到的问题在集群模式下往往出现，下面将结合实际详细介绍在 CentOS-6.x 系统上 hadoop-2.2.0 的集群安装（其他Linux发行版无太大差别），最后运行WordCount程序以验证Hadoop集群安装是否成功。<!--more-->

## 机器准备

假设集群中有三台机器，机器可以为三台物理机或虚拟机，保证三台机器可以互相通信，其中一台机器作为master（运行NameNode和ResourceManager），另外两台机器作为slave或worker（运行DataNode和NodeManager）。下面我准备的机器相关配置如下，注意每台机器要保证用户名一致。

| 主机名 |	用户名	|  IP 地址 |
|:-------:|:-------:|:--------------:|
| master	| hadoop	| 192.168.100.10 |
| slave1	| hadoop	| 192.168.100.11 |
| slave2	| hadoop	| 192.168.100.12 |

## 工具准备

为了避免在三台机器中重复安装配置工作，我们可以只在master机器上做安装配置，然后直接将配置好的软件打包发到每台slave机器上解压即可，首先我们应配置master机器到其他机器ssh免密码登陆，这是所有后续安装工作的前提。

### 1. 配置host

在master机器中配置host，在/etc/hosts文件中添加以下配置：

    192.168.100.10     master
    192.168.100.11     slave1
    192.168.100.12     slave2

### 2. 配置master免密码登录

首先运行如下命令生成公钥：

    [hadoop@master ~]$ ssh-keygen -t  rsa

将公钥拷贝到每台机器中，包括本机，以使得ssh localhost免密码登录：

    [hadoop@master ~]$ ssh-copy-id -i ~/.ssh/id_rsa.pub  hadoop@master
    [hadoop@master ~]$ ssh-copy-id -i ~/.ssh/id_rsa.pub  hadoop@slave1
    [hadoop@master ~]$ ssh-copy-id -i ~/.ssh/id_rsa.pub  hadoop@slave2

为了更好的管理集群，切换到root身份，重复上述ssh无密码设置过程，保证root身份也无能密码登录：

    [root@master ~]$ su root
    [root@master ~]$ ssh-keygen -t  rsa
    [root@master ~]$ ssh-copy-id -i ~/.ssh/id_rsa.pub  root@master
    [root@master ~]$ ssh-copy-id -i ~/.ssh/id_rsa.pub  root@slave1
    [root@master ~]$ ssh-copy-id -i ~/.ssh/id_rsa.pub  root@slave2

完成上述操作后，切换回hadoop用户，现在master机器可以ssh免密码的登录集群中每台机器，下面我们开始现在master机器中开始安装配置hadoop。

## JDK安装

从[oracle官网](http://www.oracle.com/technetwork/articles/javase/index-jsp-138363.html)下载jdk，放到`/home/hadoop`目录下（后续所有安装包默认安装在`/home/hadoop`目录下），我下载的版本为jdk1.7.0_40，解压后设置jdk的环境变量，环境变量最好不要设置为全局的（在/etc/profile中），只设置当前用户的环境变量即可.

    [hadoop@master ~]$ pwd
    /home/hadoop
    [hadoop@master ~]$ vim .bash_proflie
     # JAVA ENVIRONMENT
     export JAVA_HOME=$HOME/jdk1.7.0_40
     export PATH=$JAVA_HOME/bin:$PATH
     export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
    [hadoop@master ~]$ source .bash_proflie

## Hadoop安装

从[Apache官网](http://hadoop.apache.org/releases.html)下载hadoop发行版，放到`/home/hadoop`目录下，我下载的版本为hadoop-2.2.0，解压软件包后，首先设置hadoop的环境变量。

    [hadoop@master ~]$ vim .bash_proflie
     # HADOOP ENVIRONMENT
     export HADOOP_HOME=$HOME/hadoop-2.2.0
     export HADOOP_MAPRED_HOME=$HADOOP_HOME
     export HADOOP_COMMON_HOME=$HADOOP_HOME
     export HADOOP_HDFS_HOME=$HADOOP_HOME
     export YARN_HOME=$HADOOP_HOME
     export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
     export HDFS_CONF_DIR=$HADOOP_HOME/etc/hadoop
     export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop
     export HADOOP_LOG_DIR=$HADOOP_HOME/logs
    [hadoop@master ~]$ source .bash_proflie

下面我们开始配置hadoop，进入hadoop的配置目录，首先我们先在`hadoop-env.sh`和`yarn-env.sh`中设置好jdk的路径，然后开始修改hadoop相关配置文件。

### 配置hdfs

在配置文件`hdfs-site.xml`中添加以下内容。

```xml
<configuration>
  <property>
    <!-- hdfs地址 -->
    <name>fs.defaultFS</name>
    <value>hdfs://master:9000</value>
  </property>
  <property>
    <!-- hdfs中每一个block所存的份数，我这里设置1份，默认是3份 -->
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <!-- 开启hdfs web访问 -->
    <name>dfs.webhdfs.enabled</name>
    <value>true</value>
  </property>
</configuration>
```

### 配置yarn

为了能够运行MapReduce程序，需要让各个NodeManager在启动时加载shuffle server，Reduce Task通过该server从各个NodeManager上远程拷贝Map Task产生的中间结果。在配置文件`yarn-site.xml`中添加以下内容。

```xml
<configuration>
  <property>
    <name>yarn.resourcemanager.hostname</name>
    <value>master</value>
  </property>
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
  <property>
    <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
    <value>org.apache.hadoop.mapred.ShuffleHandler</value>
  </property>
</configuration>
```

### 配置MapReduce计算框架

为了利用MapReduce中的WordCount验证hadoop集群是否安装成功，需要为hadoop配置MapReduce计算框架。在配置文件`mapred-site.xml`中添加以下内容。

```xml
<configuration>
  <property>
    <!-- 指定yarn为MapReduce的资源调度平台 -->
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
</configuration>
```

### 配置slaves

在配置文件`slaves`中添加以下内容。

    slave1
    slave2

到这里为止，我们已经完成master机器上hadoop的配置，更多关于hadoop的配置参数说明请查看[官方文档]((http://hadoop.apache.org/docs/r2.2.0/))左侧的Configuration一栏，下面我们要将master机器上所有的安装配置操作同步到集群中所有节点上，为了避免挨个节点的重复劳动，我们前面也设置好了ssh无密码登录，现在我们简单写几个脚本，完成同步安装操作。

### 同步配置

首先同步/etc/hosts文件，这个需要切换到root用户下来完成，运行如下脚本：

```shell
#! /bin/bash
for SLAVE in `cat /home/hadoop/hadoop-2.2.0/etc/hadoop/slaves`; do
  if test `expr match $SLAVE "*slave*"` -eq 0; then
    rsync /etc/hosts $SLAVE:/etc/hosts
    #清空防火墙配置
    iptables -F
  fi
done
```

其中rsync为文件同步命令，每次配置作修改后可以通过rsync命令进行文件的同步操作。完成上述操作后，切换回hadoop用户下，运行如下脚本，完成hadoop的同步安装：

```shell
#! /bin/bash

for SLAVE in `cat $HADOOP_CONF_DIR/slaves`; do
  if test `expr match $SLAVE "*slave*"` -eq 0; then
    #拷贝jdk
    scp /home/hadoop/jdk-7u40-linux-i586.gz  hadoop@$SLAVE:/home/hadoop/
    #拷贝hadoop
    scp /home/hadoop/hadoop-2.2.0.tar.gz     hadoop@$SLAVE:/home/hadoop/
    #拷贝环境变量配置
    scp /home/hadoop/.bash_profile           hadoop@$SLAVE:/home/hadoop/

    echo "set up ${SLAVE}..."
    ssh $SLAVE "cd /home/hadoop;
                tar zxvf jdk-7u40-linux-i586.gz;
                tar zxvf hadoop-2.2.0.tar.gz;
                source /home/hadoop/.bash_profile;
                exit"
    rsync -r /home/hadoop/hadoop-2.2.0/etc/hadoop $SLAVE:/home/hadoop/hadoop-2.2.0/etc/hadoop
  fi
done
```

每次修改hadoop配置后，运行如下脚本同步到集群所有节点：

```shell
#! /bin/bash
for SLAVE in `cat $HADOOP_CONF_DIR/slaves`; do
  if test `expr match $SLAVE "*slave*"` -eq 0; then
    rsync -r /home/hadoop/hadoop-2.2.0/etc/hadoop $SLAVE:/home/hadoop/hadoop-2.2.0/etc/hadoop
  fi
done
```

到这里，hadoop的集群安装就完成了，我们启动hadoop环境，通过WordCount来测试一下集群，一次运行如下命令以启动hdfs和yarn。

## WordCount测试

在master集群上启动hdfs和yarn后台进程：

    [hadoop@master ~]$ $HADOOP_HOME/bin/hadoop namenode -format
    [hadoop@master ~]$ $HADOOP_HOME/sbin/start-dfs.sh
    [hadoop@master ~]$ $HADOOP_HOME/sbin/start-yarn.sh

启动后可以通过 http://master:50070 地址访问hdfs，通过 http://master:8088 地址访问yarn，如果不能访问，请检查master机器上的iptables配置，将相关配置清除。另外，在启动集群过程中可能会遇到DataNode、NodeManager启动不起来的现象，这也可能是slave机器上的防火墙配置导致集群间RPC通信失败的缘故，比较暴力的做法是把集群中所有机器的防火墙服务直接给停掉，相关iptables防火墙操作请自行google，下面我们运行hadoop自带的WordCount实例来验证hadoop是否能正常工作。

首先准备输入文件，上传到hdfs：

    [hadoop@master ~]$ mkdir input
    [hadoop@master ~]$ sh -c 'echo "hello hadoop" > input/f1.txt'
    [hadoop@master ~]$ sh -c 'echo "hello java" > input/f2.txt'
    [hadoop@master ~]$ sh -c 'echo "hello world" > input/f3.txt'
    [hadoop@master ~]$ $HADOOP_HOME/bin/hdfs -mkdir /input
    [hadoop@master ~]$ $HADOOP_HOME/bin/hdfs -put input/* /input/

运行WordCount Mapreduce实例：

    [hadoop@master ~]$ $HADOOP_HOME/bin/hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-examples-2.2.0.jar wordcount /input /output

作业提交后，可以访问web ui页面，观察MapReduce作业的运行情况。
