"""Single-body observer for ATANOR Contract/Body.raw(). No imports from ATANOR."""
import json
from math import prod
import numpy as np
from .organ import LokiMemory


class AtanorImageMemory:
    def __init__(self, contract, budget=262144, image_index=None):
        images = [i for i, sensor in enumerate(contract.sensors) if sensor.kind == "image"]
        if image_index is None:
            if len(images) != 1:
                raise ValueError("select an image_index when the contract has multiple or zero images")
            image_index = images[0]
        if image_index not in images:
            raise ValueError("not an image channel")
        self.index = image_index
        self.shape = tuple(contract.sensors[image_index].shape)
        self.actions = prod(contract.motors)
        self.memory = LokiMemory(prod(self.shape), budget)

    def _binding(self, wiring_id):
        # A stable opaque ID distinguishes bodies with equal-sized, permuted levers.
        if not isinstance(wiring_id, str) or not wiring_id:
            raise ValueError("supply the stable ID of this body wiring")
        return json.dumps([wiring_id, self.index, self.shape, self.actions])

    def save(self, path, wiring_id):
        self.memory.save(path, binding=self._binding(wiring_id))

    def restore(self, path, wiring_id):
        recovered = LokiMemory.load(path, binding=self._binding(wiring_id))
        if recovered.width != self.memory.width or recovered.budget != self.memory.budget:
            raise ValueError("checkpoint dimensions or budget changed")
        self.memory = recovered

    def step(self, raw, previous_action=None, body=0, cut=False, available=None):
        """Pass the already acquired raw bundle. Returns reconstructed HWC float [0,1].

        Use a separate observer per body. A cut discards the incoming action.
        The upstream room uses an all-zero image for darkness. Pass available
        explicitly if black is a legitimate observation on another body.
        """
        if previous_action is not None:
            if hasattr(previous_action, "item"):
                previous_action = previous_action.item()
            self.memory._check_action(previous_action)
            if previous_action >= self.actions:
                raise ValueError("action outside contract")
        frame = raw[self.index][body]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.shape != self.shape:
            raise ValueError("body image does not match the contract")
        if cut:
            self.memory.cut()
            previous_action = None
        sensed = frame.any() if available is None else available
        if sensed:
            self.memory.observe(frame.astype(np.float32).ravel() / 255, previous_action)
            return self.memory.imagine(()).reshape(self.shape)
        if previous_action is None:
            return self.memory.imagine(()).reshape(self.shape)
        return self.memory.advance(previous_action).reshape(self.shape)
