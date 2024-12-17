from abc import ABCMeta, abstractmethod

class CVModel(metaclass = ABCMeta):
    # methods we expect to be overridden
    @abstractmethod
    def train_model(self, train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader, **kwargs) -> dict[str, Any]:
        pass
    
    @abstractmethod
    def validate_model(self, loader: torch.utils.data.DataLoader, **kwargs) -> dict[str, Any]:
        pass
    
    @abstractmethod
    def test_model(self, loader: torch.utils.data.DataLoader, **kwargs) -> dict[str, Any]:
        pass
    
    @abstractmethod
    def predict(self, loader: torch.utils.data.DataLoader, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def save(self, path: Path | str, **kwargs) -> None:
        pass
    
    @abstractmethod
    def interpret(self, test_input: torch.utils.data.DataLoader, **kwargs) -> None:
        pass