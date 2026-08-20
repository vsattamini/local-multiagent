# tests/test_project_structure.py
def test_project_modules_import():
    from src.agents import base
    from src.models import interface
    from src.coordination import coordinator
    from src.utils import memory
    assert True
