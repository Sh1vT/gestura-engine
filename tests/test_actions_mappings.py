import os
import json
import tempfile
from app.utils import load_actions, save_actions, load_mappings, save_mappings

def test_save_and_load_actions():
    actions = [
        {"name": "switch", "type": "macro", "params": {"keys": "alt+tab"}},
        {"name": "fullScreen", "type": "macro", "params": {"keys": "f"}}
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "actions.json")
        save_actions(test_file, actions)
        loaded = load_actions(test_file)
        assert loaded == actions

def test_save_and_load_mappings():
    mappings = [
        {"sign": "thumbsUp", "action": "fullScreen"},
        {"sign": "palmOpen", "action": "switch"}
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "map.json")
        save_mappings(test_file, mappings)
        loaded = load_mappings(test_file)
        assert loaded == mappings

def test_load_actions_missing_file():
    assert load_actions("/tmp/nonexistent_actions_file.json") == []

def test_load_mappings_missing_file():
    assert load_mappings("/tmp/nonexistent_mappings_file.json") == [] 