---
title: 临时更换hadoop-ugi
categories: 问题总结
date: 2016-05-12 21:45:11
comments: false
tags:
  - hadoop
  - ugi
  - hdfs
---


在用spark读写hdfs数据时，有时候当前用户对要读写的hdfs路径没有权限，需要临时改变用户去读写hdfs，操作完后回到原来的用户。我们的hdfs是没有权限认证的，一开始通过下面代码的方式来实现。<!--more-->

```scala
val conf = new JobConf(rdd.context.hadoopConfiguration)
conf.set("hadoop.job.ugi", "newuser,newgroup")
rdd.saveAsHadoopFile(path, keyClass, valueClass, outputFormatClass, conf)
```

以上代码是在现有的 hadoop conf 上复制出一个 conf，然后修改ugi，利用这个新的 conf 去写数据，表面上看并没有什么错，没有改变原来的 conf，然而这种方式偶尔会不生效。经过排查发现，自从hdfs-2.0开始，ugi操作需要通过`UserGroupInformation`这个类来实现，在获取用户名时有个缓存，所以之前偶尔不生效的情况是因为它用了缓存里的用户名而还没来得及更新，为了规避这个问题，改成下面的实现就万无一失了，所有临时的操作在`doAs`方法里实现，出了这个方法就回到原来用户下。

```scala
val conf = new JobConf(rdd.context.hadoopConfiguration)
conf.set("hadoop.job.ugi", "newuser,newgroup")
val ugi = UserGroupInformation.getCurrentUser(conf)
ugi.doAs(new PrivilegeAction[unit] {
  def run(){
    rdd.saveAsHadoopFile(path, keyClass, valueClass, outputFormatClass, conf)
  }
})
```
