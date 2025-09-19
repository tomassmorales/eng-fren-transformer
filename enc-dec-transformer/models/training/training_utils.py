import tensorflow as tf
import math

class WarmUpThenDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.base_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        return self.base_lr * tf.math.rsqrt(tf.cast(step, tf.float32))

def create_optimizer(num_examples, BATCH_SIZE, num_epochs, warmup_ratio=0.10):
    steps_per_epoch = math.ceil(num_examples / BATCH_SIZE)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    base_lr = 1e-4
    lr_schedule = WarmUpThenDecay(base_lr, warmup_steps, total_steps)
    
    return tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)