import pytest
from src.models.interface import ModelInterface
from src.models.manager import ModelManager

@pytest.mark.asyncio
async def test_model_manager_initialization():
    manager = ModelManager()
    assert manager.loaded_models == {}
    
@pytest.mark.asyncio
async def test_model_loading():
    manager = ModelManager()
    model = await manager.load_model("test-model", "path/to/model")
    assert model is not None
    assert "test-model" in manager.loaded_models
