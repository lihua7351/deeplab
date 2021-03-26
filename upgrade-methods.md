
## 环境：win10,tensorflow-gpu==2.0.0/2.4.1

### 转换工具

#### tf_upgrade_v2

##### win10下使用方法：整个根目录转换


```python
tf_upgrade_v2  ^
  --intree E:/name/filepath/models-research/deeplab/  ^
  --outtree deeplab_v2  ^
  --reportfile deeplab-report.txt
```

##### win10下使用方法：单个文件转换


```python
tf_upgrade_v2  ^
--infile crop_to_seg.py  ^
--outfile crop_to_seg-upgraded.py  ^
--reportfile crop_to_seg-report.txt
```

#### 转换期间遇到的一些问题：UnicodeDecodeError:"gbk" codec can't decode byte ...
#### 解决方法：把文档里面的所有中文删除（包含中文注释和含有中文的路径等）

#### 转换结果


```python
tf.variable_scope
==>>tf.compat.v1.variable_scope

tf.logging.info
==>>tf.compat.v1.logging.info

tf.placeholder
==>>tf.compat.v1.placeholder

tf.convert_to_tensor(inputs)
==>>tf.convert_to_tensor(value=inputs)

tf.shape(net)
==>>tf.shape(input=net)

tf.zeros_initializer()
==>>tf.compat.v1.zeros_initializer()

tf.nn.moments(self.kernel, [0, 1, 2], keep_dims=True)
==>>tf.nn.moments(x=self.kernel, axes=[0, 1, 2], keepdims=True)

tf.image.resize_bilinear(pooled_features,current_config[_TARGET_SIZE],align_corners=True)
==>>tf.image.resize(pooled_features,current_config[_TARGET_SIZE],method=tf.image.ResizeMethod.BILINEAR)

tf.image.resize(resized_label,new_size,method=get_label_resize_method(label),align_corners=align_corners)
==>>tf.image.resize(resized_label,new_size,method=get_label_resize_method(label))

tf.lin_space
==>>tf.linspace

tf.random_shuffle
==>>tf.random.shuffle

tf.gfile.Glob
==>>tf.io.gfile.glob

tf.python_io.TFRecordWriter
==>>tf.io.TFRecordWriter

tf.gfile.MakeDirs
==>>tf.io.gfile.makedirs

tf.FixedLenFeature
==>>tf.io.FixedLenFeature

return dataset.make_one_shot_iterator()
==>>return tf.compat.v1.data.make_one_shot_iterator(dataset)

tf.train.write_graph
==>>tf.io.write_graph

tf.reverse_v2
==>>tf.reverse
```

#### 部分手动修改


```python
tf.to_float(inputs)
==>>tf.cast(inputs, dtype=dtype) ## dtype=tf.float32

self.assertRaisesWithRegexpMatch
==>>self.assertRaisesRegex
```

### 接口转换方法：

#### 1：tensorflow2.x不支持tensorflow.contrib模块,需将其全部注释替换


```python
from tensorflow.contrib import framework as contrib_framework
>contrib_framework.add_arg_scope
>>arg_scope = contrib_framework.arg_scope
>>>contrib_framework.get_variables_to_restore
>>>>contrib_framework.assign_from_checkpoint
==>>
>from tf_slim.ops.arg_scope import add_arg_scope
>>import tf_slim as slim
>>arg_scope = slim.arg_scope
>>>slim.get_variables_to_restore
>>>>slim.assign_from_checkpoint


from tensorflow.contrib import layers as contrib_layers
contrib_layers.xavier_initializer()
==>>from tf_slim.layers import initializers
initializers.xavier_initializer()

from tensorflow.contrib import layers as contrib_layers
contrib_layers.l2_regularizer(weight_decay)
contrib_layers.variance_scaling_initializer(factor=1 / 3.0, mode='FAN_IN', uniform=True)
==>>import tf_slim as slim
slim.l2_regularizer(weight_decay)
slim.variance_scaling_initializer(factor=1 / 3.0, mode='FAN_IN', uniform=True)

from tensorflow.contrib.layers.python.layers import layers
==>>from tf_slim.layers import layers

from tensorflow.contrib.layers.python.layers import utils
==>>from tf_slim.layers import utils

from tensorflow.contrib.slim.nets import resnet_utils
==>>from tf_slim.nets import resnet_utils

from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim
==>>import tf_slim as slim

from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim
dataset = slim.dataset
==>>import tf_slim as slim
dataset = slim.data.dataset

from tensorflow.contrib import training as contrib_training
contrib_training.HParams
contrib_training.evaluate_repeatedly
contrib_training.SummaryAtEndHook
contrib_training.checkpoints_iterator
==>>from tensorboard.plugins.hparams import api as hp
hp.hparams
import tf_slim as slim
slim.evaluation.evaluation_loop
from tf_slim.training import evaluation
evaluation.SummaryAtEndHook
tf.train.checkpoints_iterator

from tensorflow.contrib import metrics as contrib_metrics
contrib_metrics.aggregate_metric_map
==>>import tf_slim as slim
slim.metrics.aggregate_metric_map

from tensorflow.contrib import tfprof as contrib_tfprof
contrib_tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),
        tfprof_options=contrib_tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
==>>tf.profiler.profile(tf.compat.v1.get_default_graph(),
        tfprof_options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())

from tensorflow.contrib import tfprof as contrib_tfprof
contrib_tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),
        tfprof_options=contrib_tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
==>>tf.profiler.profile(tf.compat.v1.get_default_graph(),
        tfprof_options=tf.profiler.ProfileOptionBuilder.float_operation())

from tensorflow.contrib import tfprof as contrib_tfprof
with contrib_tfprof.ProfileContext(enabled=profile_dir is not None, profile_dir=profile_dir)
==>> from tensorflow_core.python.eager import profiler
with profiler.Profiler(profile_dir)
```
