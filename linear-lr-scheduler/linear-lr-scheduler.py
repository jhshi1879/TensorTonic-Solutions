def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
  """
  Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
  Steps are 0-based; clamp at final_lr after total_steps.
  """
  # Write code here
  # if warmup_steps == 0:
  #   return lr if step < total_steps else final_lr
  if total_steps == 0:
    return final_lr

  if step < warmup_steps and warmup_steps > 0:
    lr = step * initial_lr / warmup_steps
  if warmup_steps <= step <= total_steps and total_steps != warmup_steps:
      lr = final_lr + (initial_lr - final_lr) * (total_steps - step) / (total_steps - warmup_steps)
  if step > total_steps:
    lr = final_lr
  return lr