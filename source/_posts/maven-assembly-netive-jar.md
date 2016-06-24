---
title: Maven 使用本地jar包
tags:
  - maven
  - java
categories: 问题总结
date: 2016-06-24 21:45:11
---

通常maven管理的项目中的依赖都是在远程仓库中的，假如我需要在maven项目中添加一个本地的jar包依赖，该jar包在仓库中是不存在的，可能是项目组前人开发的一个库，但是没发布到maven仓库中。遇到这种情况我们可以通过在pom中指定本地的依赖，如：
```xml
<dependency>
    <groupId>com.tencent.ml.tools</groupId>
    <artifactId>tools</artifactId>
    <version>0.0.1</version>
    <scope>system</scope>
    <systemPath>${project.basedir}/lib/tools-0.0.1.jar</systemPath>
</dependency>
```

通过以上方法仅仅可以让你的代码编译通过，如果我们要把这个本地jar包打进assembly jar包，那么需要在pom中添加如下配置：
```xml
<build>
  <plugins>
    <plugin>
      <groupId>org.apache.maven.plugins</groupId>
      <artifactId>maven-assembly-plugin</artifactId>
      <version>2.5.3</version>
      <configuration>
        <descriptorRefs>
          <descriptorRef>jar-with-dependencies</descriptorRef>
        </descriptorRefs>
      </configuration>
      <executions>
        <execution>
          <phase>package</phase>
          <goals>
            <goal>single</goal>
          </goals>
        </execution>
      </executions>
    </plugin>
  </plugins>
  <resources>
    <resource>
      <directory>lib/</directory>
      <includes>
        <include>tools-0.0.1.jar</include>
      </includes>
    </resource>
  </resources>
</build>
```
