
## 环境：win10,tensorflow-gpu==2.0.0/2.4.1

### 转换工具

#### tf_upgrade_v2

##### win10下使用方法：整个根目录转换


```python
tf_upgrade_v2  ^
  --intree E:/LGH/0311/models-research/deeplab/  ^
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

tf.convert_to_tensor(inputs)
==>>tf.convert_to_tensor(value=inputs)

tf.zeros_initializer()
==>>tf.compat.v1.zeros_initializer()

tf.nn.moments(self.kernel, [0, 1, 2], keep_dims=True)
==>>tf.nn.moments(x=self.kernel, axes=[0, 1, 2], keepdims=True)
```

### 接口转换方法：

#### 1：tensorflow2.x不支持tensorflow.contrib模块,需将其全部注释


```python
from tensorflow.contrib import framework as contrib_framework
contrib_framework.add_arg_scope
==>>from tf_slim.ops.arg_scope import add_arg_scope

from tensorflow.contrib import layers as contrib_layers
contrib_layers.xavier_initializer()
==>>from tf_slim.layers import initializers
initializers.xavier_initializer()

from tensorflow.contrib.layers.python.layers import layers
==>>from tf_slim.layers import layers

from tensorflow.contrib.layers.python.layers import utils
==>>from tf_slim.layers import utils

from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim
==>>import tf_slim as slim


```
