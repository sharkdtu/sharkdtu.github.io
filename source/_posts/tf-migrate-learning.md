---
title: TensorFlow 迁移学习实践小记
date: 2020-11-32 21:23:01
categories: 机器学习
comments: false
tags:
  - TensorFlow
---

在我们的很多推荐业务场景中，通常一个模型可能是一直不断增量训练的，如果哪天业务需要调整模型结构，去训练一个新模型，但是又不想完全从0开始，希望复用原来模型里面的部分参数，这样冷启动的代价就小很多了。<!--more-->

实际上 TensorFlow 提供了足够的灵活性，我们可以控制从其他模型 restore 部分参数到新的模型里。因为目前生产环境普遍还是在用 tf-1.x，下面分别介绍Low-Level API 和 Estimator API 两种实践。

## Low-Level API 实践

在决定从已有的模型预热参数前，可以先将模型ckpt拉到本地，开一个 ipython 或 jupyter，列出模型中的所有参数。
```python
In [6]: tf.train.list_variables(checkpoint_dir)
Out[6]:
[('dense/bias', [1]),
 ('dense/bias/Adagrad', [1]),
 ('dense/kernel', [17, 1]),
 ('dense/kernel/Adagrad', [17, 1]),
 ('fm/b', [1]),
 ('fm/b/Adagrad', [1]),
 ('fm/v', [4809162, 16]),
 ('fm/v/Adagrad', [4809162, 16]),
 ('fm/w', [4809162, 1]),
 ('fm/w/Adagrad', [4809162, 1]),
 ('global_step', [])]
```

假如，我们想要从ckpt中预热 `fm/v` 和 `fm/w` 两个参数，很简单，通过自定义一个 `tf.train.Saver` 来控制加载哪些参数：

```python
...
w2 = tf.get_variable(
    "w2", shape=[4809162, 1],
    dtype=tf.float32,
    initializer=tf.initializers.zeros())
v2 = tf.get_variable(
    "v2",
    shape=[4809162, 16],
    dtype=tf.float32,
    initializer=tf.initializers.truncated_normal(mean=0.0, stddev=1 / math.sqrt(16)))

ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)
recover_vars = {'fm/w': w2, 'fm/v': v2}
saver = tf.train.Saver(recover_vars)
...

with tf.Session() as sess:
  ...
  sess.run(init_op)
  saver.restore(sess, ckpt_state.model_checkpoint_path)
  ...
```

以上代码中 `recover_vars` 定义了要从 ckpt 中恢复的参数，是一个字典形式，key 为 ckpt 中的变量名，从上面我们 list 出来的变量里找即可，value 为要覆盖的变量，即从 ckpt 中找到名字为 key 的变量参数，去覆盖 value 指定的变量。

## Estimator API 实践

如果你是用高阶 Estimator API，其实完全可以借助 Estimator 自带的 [warm_start](https://www.tensorflow.org/api_docs/python/tf/estimator/WarmStartSettings) 功能来实现。

```python
tf.estimator.WarmStartSettings(
    ckpt_to_initialize_from, vars_to_warm_start='.*', var_name_to_vocab_info=None,
    var_name_to_prev_var_name=None
)
```

* ckpt_to_initialize_from：预热模型的ckpt路径
* vars_to_warm_start：要加载哪些变量出来预热，可以通过上述 `tf.train.list_variables` 方法先列出变量名再决定要哪些变量
* var_name_to_vocab_info：动态词表信息
* var_name_to_prev_var_name：新模型中的变量名 -> 旧模型中的变量名，意思就是加载出来的变量会预热到新模型的变量


如果旧模型中有变量A，新模型有变量A、B，需要将旧模型的变量A恢复到新模型的变量B，如果使用tf warm_start，它既会将旧模型的变量A恢复到新模型的变量B，也会恢复到新模型的变量A。为解决名字冲突问题，我们可以自定义一个 Hook 将上述 low-level api 的使用方式封装一下，实现定制化恢复即可。

```python
class VariableRecoverHook(tf.train.SessionRunHook):
  """Recover specified variables from checkpoint."""

  def __init__(self, ckp_dir, recover_vars):
    """Initializes a `VariableRecoverHook`.

    Args:
      ckp_dir: Checkpoint directory where variables recover from
      recover_vars: A `dict` of names to variables
    """
    self._ckpt_state = tf.train.get_checkpoint_state(ckp_dir)
    if not isinstance(recover_vars, dict):
      raise ValueError("recover_vars must be a dict of names to variables")
    self._vars = recover_vars

  def begin(self):
    """Create a tf saver for recover variables."""
    self._saver = tf.train.Saver(self._vars)

  def after_create_session(self, session, coord):
    """Recover variables from checkpoint."""
    self._saver.restore(session, self._ckpt_state.model_checkpoint_path)
```

以上代码实现一个 Hook，其中初始化参数 `recover_vars` 表示要从 ckpt 中恢复的参数。一般在恢复参数前，也需要list一下旧模型中的参数，找到对应的变量名。有了这个 Hook 后，那么我们就可以在 `model_fn` 中插入这个 Hook 的实例即可。

```python
def model_fn(features, labels, mode, params):
  ...
  spec = head.create_estimator_spec(...)
  recover_hook = VariableRecoverHook(ckp_dir=old_ckpt_dir, recover_vars=recover_vars)
  return spec._replace(training_hooks=(spec.training_hooks + (recover_hook,)))
```


## 小结

实践中我们可以基于 TensorFlow 灵活保存以及恢复参数，当有迁移学习需求时，可以通过定制化 `tf.train.Saver` 的方式来控制预热指定的参数。目前 TensorFlow 也进入 2.x 时代了，官方主推Keras API，通过 Keras API 可以更加灵活的控制[保存以及恢复参数](https://www.tensorflow.org/guide/checkpoint)。但是如果你是用 Estimator，则可以直接复用 warm_start 或上述 Hook 实现。
