from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
_HOOKS = "hooks"
_STEPS_PER_RUN_VAR = "steps_per_run"



class NPYHook(tf.train.LoggingTensorHook):
  def after_run(self, run_context, run_values):
    _ = run_context
    if self._should_trigger:
      values = np.array(run_values.results)

      np.save(str('values') + '.npy', values)


    self._iter_count += 1
