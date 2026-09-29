from pathlib import Path


def test_repository_root_contains_only_directories_and_agent_guide():
    root = Path(__file__).resolve().parents[1]

    assert {path.name for path in root.iterdir() if path.is_file()} == {"AGENTS.md", "README.md"}
