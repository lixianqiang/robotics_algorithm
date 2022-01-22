from abc import ABC, abstractmethod


class IStrategy(ABC):
    """策略模式——策略类"""

    @abstractmethod
    def func(self, arg1, arg2):
        pass

    @abstractmethod
    def grad_func(self, arg1, arg2):
        pass


class Context:
    """策略模式——上下文类"""

    def __init__(self, strategy: IStrategy = None):
        self._strategy = strategy

    def ChangeStrategy(self, strategy: IStrategy):
        self._strategy = strategy

    def func(self, arg1, arg2):
        return self._strategy.func(arg1, arg2)

    def grad_func(self, arg1, arg2):
        return self._strategy.grad_func(arg1, arg2)