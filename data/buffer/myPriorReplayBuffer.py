from typing import Any, Union, List, Tuple, Optional
import numpy as np
import torch
from tianshou.data import Batch
from tianshou.data.batch import _is_scalar, _alloc_by_keys_diff
from tianshou.data.buffer.prio import PrioritizedReplayBuffer


def _create_value(
    inst: Any,
    size: int,
    stack: bool = True,
) -> Union["Batch", np.ndarray, torch.Tensor]:
    """Create empty place-holders accroding to inst's shape.

    :param bool stack: whether to stack or to concatenate. E.g. if inst has shape of
        (3, 5), size = 10, stack=True returns an np.ndarry with shape of (10, 3, 5),
        otherwise (10, 5)
    """
    # has_shape = isinstance(inst, (np.ndarray, torch.Tensor))
    is_scalar = _is_scalar(inst)
    if not stack and is_scalar:
        # should never hit since it has already checked in Batch.cat_ , here we do not
        # consider scalar types, following the behavior of numpy which does not support
        # concatenation of zero-dimensional arrays (scalars)
        raise TypeError(f"cannot concatenate with {inst} which is scalar")
    # if has_shape:
    #     shape = (size, *inst.shape) if stack else (size, *inst.shape[1:])
    if isinstance(inst, np.ndarray):
        # target_type = inst.dtype.type if issubclass(inst.dtype.type,
        #                                             (np.bool_,
        #                                              np.number)) else object
        return np.array([None for _ in range(size)], object)
    elif isinstance(inst, torch.Tensor) and not inst.is_sparse:
        # rewrite the class PriorReplayBufffer for sparseTensor
        return np.array([None for _ in range(size)], object)
    elif isinstance(inst, (dict, Batch)):
        zero_batch = Batch()
        for key, val in inst.items():
            zero_batch.__dict__[key] = _create_value(val, size, stack=stack)
        return zero_batch
    elif is_scalar:
        return _create_value(np.asarray(inst), size, stack=stack)
    else:  # fall back to object
        return np.array([None for _ in range(size)], object)


class MyPriorReplayBuffer(PrioritizedReplayBuffer):

    def add(
        self,
        batch: Batch,
        buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # preprocess batch
        new_batch = Batch()
        for key in set(self._reserved_keys).intersection(batch.keys()):
            new_batch.__dict__[key] = batch[key]
        batch = new_batch
        assert set(["obs", "act", "rew", "done"]).issubset(batch.keys())
        stacked_batch = buffer_ids is not None
        if stacked_batch:
            assert len(batch) == 1
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1] if stacked_batch else batch.obs[-1]
        if not self._save_obs_next:
            batch.pop("obs_next", None)
        elif self._save_only_last_obs:
            batch.obs_next = (batch.obs_next[:, -1]
                              if stacked_batch else batch.obs_next[-1])
        # get ptr
        if stacked_batch:
            rew, done = batch.rew[0], batch.done[0]
        else:
            rew, done = batch.rew, batch.done
        ptr, ep_rew, ep_len, ep_idx = list(
            map(lambda x: np.array([x]), self._add_index(rew, done)))
        try:
            if len(batch) == 1:
                ptr = ptr.item()
        except TypeError:
            ptr = ptr.item()
        try:
            self._meta[ptr] = batch
        except ValueError:
            stack = not stacked_batch
            batch.rew = batch.rew.astype(float)
            batch.done = batch.done.astype(bool)
            if self._meta.is_empty():
                self._meta = _create_value(  # type: ignore
                    batch, self.maxsize, stack)
            else:  # dynamic key pops up in batch
                _alloc_by_keys_diff(self._meta, batch, self.maxsize, stack)
            self._meta[ptr] = batch
        self.init_weight(ptr)
        return ptr, ep_rew, ep_len, ep_idx
