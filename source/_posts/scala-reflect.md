---
title: 初识Scala反射
date: 2016-04-23 10:04:25
categories: 程序语言
comments: false
tags:
  - 反射
  - scala
---
我们知道，scala编译器会将scala代码编译成JVM字节码，编译过程中会擦除scala特有的一些类型信息，在scala-2.10以前，只能在scala中利用java的反射机制，但是通过java反射机制得到的是只是擦除后的类型信息，并不包括scala的一些特定类型信息。从scala-2.10起，scala实现了自己的反射机制，我们可以通过scala的反射机制得到scala的类型信息。scala反射包括运行时反射和编译时反射，本文主要阐述运行时反射的一些用法，方便scala开发人员参考，具体原理细节请查看官方文档。<!--more-->本文涉及到的代码示例是基于scala-2.10.4，如有不同请勿对号入座。

给定类型或者对象实例，通过scala运行时反射，可以做到：1）获取运行时类型信息；2）通过类型信息实例化新对象；3）访问或调用对象的方法和属性等。下面分别举例阐述运行时反射的功能。

## 获取运行时类型信息

scala运行时类型信息是保存在`TypeTag`对象中，编译器在编译过程中将类型信息保存到`TypeTag`中，并将其携带到运行期。我们可以通过`typeTag`方法获取`TypeTag`类型信息。

    scala> import scala.reflect.runtime.universe._
    import scala.reflect.runtime.universe._

    scala> typeTag[List[Int]]
    res0: reflect.runtime.universe.TypeTag[List[Int]] = TypeTag[scala.List[Int]]

    scala> res0.tpe
    res1: reflect.runtime.universe.Type = scala.List[Int]


如上述scala REPL显示，通过`typeTag`方法获取`List[Int]`类型的`TypeTag`对象，该对象包含了`List[Int]`的详细类型信息，通过`TypeTag`对象的`tpe`方法得到由`Type`对象封装具体的类型信息，可以看到该`Type`对象的类型信息精确到了类型参数`Int`。如果仅仅是获取类型信息，还有一个更简便的方法，那就是通过`typeOf`方法。

    scala> import scala.reflect.runtime.universe._
    import scala.reflect.runtime.universe._

    scala> typeOf[List[Int]]
    res0: reflect.runtime.universe.Type = scala.List[Int]

这时有人就会问，`typeTag`方法需要传一个具体的类型，事先知道类型还要`TypeTag`有啥用啊。我们不妨写个方法，获取任意对象的类型信息。

    scala> val ru = scala.reflect.runtime.universe
    ru: scala.reflect.api.JavaUniverse = scala.reflect.runtime.JavaUniverse@6dae22e7

    scala> def getTypeTag[T: ru.TypeTag](obj: T) = ru.typeTag[T]
    getTypeTag: [T](obj: T)(implicit evidence$1: ru.TypeTag[T])ru.TypeTag[T]

    scala> val list = List(1, 2, 3)
    list: List[Int] = List(1, 2, 3)

    scala> val theType = getTypeTag(list).tpe
    theType: ru.Type = List[Int]

如上scala REPL显示，方法`getTypeTag`可以获取任意对象的类型信息，注意方法中的上下文界定`T: ru.TypeTag` ,它表示存在一个从`T`到`TypeTag[T]`的隐式转换，前面已经讲到，`TypeTag`对象是在编译期间由编译器生成的，如果不加这个上下文界定，编译器就不会为T生成`TypeTag`对象。当然也可以通过隐式参数替代上下文界定，就如同REPL中显示的`implicit evidence$1: ru.TypeTag[T]`那样。一旦我们获取到了类型信息(Type instance)，我们就可以通过该`Type`对象查询更详尽的类型信息。

    scala> val decls = theType.declarations.take(10)
    decls: Iterable[ru.Symbol] = List(constructor List, method companion, method isEmpty, method head, method tail, method ::, method :::, method reverse_:::, method mapConserve, method ++)

到这里我们知道`TypeTag`对象封装了`Type`对象，通过`Type`对象可以获取详尽的类型信息，包括方法和属性等，通过`typeTag`方法可以得到`TypeTag`对象，通过`typeOf`方法可以得到`Type`对象，它包含没有被编译器擦除的含完整的scala类型信息，与之对应地，如果想获取擦除后的类型信息，传统的方法可以通过java的反射机制来实现，但是scala也提供了该功能，通过`classTag`方法可以获取`ClassTag`对象，`ClassTag`封装了擦除后的类型信息，通过`classOf`方法可以获取`Class`对象，这与java反射中的`Class`对象一致。

    scala> import scala.reflect._
    import scala.reflect._

    scala> val clsTag = classTag[List[Int]]
    clsTag: scala.reflect.ClassTag[List[Int]] = scala.collection.immutable.List

    scala> clsTag.runtimeClass
    res0: Class[_] = class scala.collection.immutable.List

    scala> val cls = classOf[List[Int]]
    cls: Class[List[Int]] = class scala.collection.immutable.List

    scala> cls.[tab键补全]
    asInstanceOf              asSubclass          cast
    desiredAssertionStatus    getAnnotation
    getAnnotations            getCanonicalName    ...   

从上述scala REPL中可以看到，`ClassTag`对象包含了`Class`对象，通过`Class`对象仅仅可以获取擦除后的类型信息，通过在scala REPL中用tab补全可以看到通过`Class`对象可以获取的信息。

## 运行时类型实例化

我们已经知道通过`Type`对象可以获取未擦除的详尽的类型信息，下面我们通过`Type`对象中的信息找到构造方法并实例化类型的一个对象。

    scala> case class Person(id: Int, name: String)
    defined class Person

    scala> val ru = scala.reflect.runtime.universe
    ru: scala.reflect.api.JavaUniverse = scala.reflect.runtime.JavaUniverse@3e9ed70d

    scala> val m = ru.runtimeMirror(getClass.getClassLoader)
    m: ru.Mirror = JavaMirror with scala.tools.nsc.interpreter.IMain$TranslatingClassLoader@a57fc5f ...

    scala> val classPerson = ru.typeOf[Person].typeSymbol.asClass
    classPerson: ru.ClassSymbol = class Person

    scala> val cm = m.reflectClass(classPerson)
    cm: ru.ClassMirror = class mirror for Person (bound to null)

    scala> val ctor = ru.typeOf[Person].declaration(ru.nme.CONSTRUCTOR).asMethod
    ctor: ru.MethodSymbol = constructor Person

    scala> val ctorm = cm.reflectConstructor(ctor)
    ctorm: ru.MethodMirror = constructor mirror for Person.<init>(id: scala.Int, name: String): Person (bound to null)

    scala> val p = ctorm(1, "Mike")
    p: Any = Person(1,Mike)

如上scala REPL代码，要想通过`Type`对象获取相关信息，必须借助`Mirror`，`Mirror`是按层级划分的，有`ClassLoaderMirror`, `ClassMirror`, `InstanceMirror`, `ModuleMirror`, `MethodMirror`, `FieldMirror`。通过`ClassLoaderMirror`可以创建`ClassMirror`, `InstanceMirror`, `ModuleMirror`,  `MethodMirror`, `FieldMirror`。通过`ClassMirror`, `InstanceMirror`可以创建`MethodMirror`,  `FieldMirror`。`ModuleMirror`用于处理单例对象，通常是由`object`定义的。从上述代码中可以发现，首先获取一个`ClassLoaderMirror`，然后通过该`Mirror`创建一个`ClassMirror`，继续创建`MethodMirror`，通过该`MethodMirror`调用构造函数。从一个`Mirror`创建另一个`Mirror`，需要指定一个`Symbol`，`Symbol`其实就是绑定名字和一个实体，有`ClassSymbol`、 `MethodSymbol`、 `FieldSymbol`等，`Symbol`的获取是通过`Type`对象方法去查询，例如上述代码中通过`declaration`方法查询构造函数的`Symbol`。

## 运行时类成员访问

下面举例阐述访问运行时类成员，同理，我们只需逐步创建`FieldMirror`来访问类成员。

    scala> case class Person(id: Int, name: String)
    defined class Person

    scala> val ru = scala.reflect.runtime.universe
    ru: scala.reflect.api.JavaUniverse = scala.reflect.runtime.JavaUniverse@3e9ed70d

    scala> val m = ru.runtimeMirror(getClass.getClassLoader)
    m: ru.Mirror = JavaMirror with scala.tools.nsc.interpreter.IMain$TranslatingClassLoader@a57fc5f ...

    scala> val p = Person(1, "Mike")
    p: Person = Person(1,Mike)

    scala> val nameTermSymb = ru.typeOf[Person].declaration(ru.newTermName("name")).asTerm
    nameTermSymb: ru.TermSymbol = value name

    scala> val im = m.reflect(p)
    im: ru.InstanceMirror = instance mirror for Person(1,Mike)

    scala> val nameFieldMirror = im.reflectField(nameTermSymb)
    nameFieldMirror: ru.FieldMirror = field mirror for Person.name (bound to Person(1,Mike))

    scala> nameFieldMirror.get
    res0: Any = Mike

    scala> nameFieldMirror.set("Jim")

    scala> p.name
    res2: String = Jim

如上代码所示，通过层级`ClassLoaderMirror`->`InstanceMirror`->`FieldMirror`得到`FieldMirror`，通过`Type`对象调用方法`declaration(ru.newTermName("name"))`获取name字段的`Symbol`，通过`FieldMirror`的`get`和`set`方法去访问和修改成员变量。

说了这么多，貌似利用java的反射机制也可以实现上述功能，还不用这么的费劲。关键还是在于你要访问编译器擦除后的类型信息还是擦除前的类型信息，如果是访问擦除后的类型信息，使用java和scala的反射都可以，但是访问擦除前的类型信息，那就必须要使用scala的反射，因为java的反射并不知道擦除前的信息。

举个栗子，一步步剖析，首先定义一个基类`A`，它包含一个抽象类型成员`T`，然后分别派生出两个子类`B`和`C`。

    scala> class A {
         | type T
         | val x: Option[T] = None
         | }
    defined class A

    scala> class B extends A
    defined class B

    scala> class C extends B
    defined class C

现在分别实例化`B`和`C`的一个对象，并将抽象类型`T`具体化为`String`类型。

    scala> val b = new B { type T = String }
    b: B{type T = String} = $anon$1@446344a8

    scala> val c = new C { type T = String }
    c: C{type T = String} = $anon$1@195bc0a4

现在通过java的反射机制判断对象`b`和`c`的运行时类型是否是父子关系。

    scala> b.getClass.isAssignableFrom(c.getClass)
    res3: Boolean = false

可以看到通过java的反射判断对象`c`的运行时类型并不是对象`b`的运行时类型的子类。然而从我们的定义来看，对象`c`的类型本应该是对象`b`的类型的子类，到这里我们就会想到编译器，在实例化对象`b`和`c`时实际上是通过匿名类来实例化的，一开始定义类型信息在编译的时候被擦除了，转为匿名类了。下面通过scala的反射机制判断对象`b`和`c`的运行时类型是否是父子关系。

    scala> val ru = scala.reflect.runtime.universe
    ru: scala.reflect.api.JavaUniverse = scala.reflect.runtime.JavaUniverse@3e9ed70d

    scala> def isSubClass[T: ru.TypeTag, S: ru.TypeTag](x: T, y: S): Boolean = {
         | val leftTag = ru.typeTag[T]
         | val rightTag = ru.typeTag[S]
         | leftTag.tpe <:< rightTag.tpe
         | }
    isSubClass: [T, S](x: T, y: S)(implicit evidence$1: ru.TypeTag[T], implicit evidence$2: ru.TypeTag[S])Boolean

    scala> isSubClass(c, b)
    res5: Boolean = true

从上述代码中可以看到，通过scala的反射得到的类型信息符合我们一开始定义。所以在scala中最好是使用scala的反射而不要使用java的反射，因为很有可能编译后通过java的反射得到的结果并不是想象的那样。
