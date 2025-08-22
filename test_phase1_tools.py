#!/usr/bin/env python3
"""
Test Suite for Phase 1 Tools: create_task, context_dumper, and prompt_providers

Tests the core tools implemented in Phase 1.3, 1.4, and 1.5
"""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import the tools to test directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from create_task import create_task
from context_dumper import dump_session_context, load_session_context
from prompt_providers import (
    root_agent_prompt_provider, 
    high_level_planner_prompt_provider,
    get_all_agent_configurations,
    validate_agent_configuration,
    load_prompt_file,
    parse_specs_file
)


class TestCreateTask:
    """Test the create_task tool functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_root = Path(__file__).parent
        
    def teardown_method(self):
        """Clean up test environment"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_task_success(self):
        """Test successful task creation"""
        with patch('create_task.get_project_root', return_value=Path(self.temp_dir)):
            result = create_task("Customer Analysis", "Analyze customer behavior and trends")
            
            assert result["success"] is True
            assert result["task_name"] == "customer_analysis"
            assert "customer_analysis" in result["task_folder"]
            assert len(result["files_created"]) == 2  # task.md and memory folder
            
            # Check that files were actually created
            task_folder = Path(self.temp_dir) / "tasks" / "customer_analysis"
            assert task_folder.exists()
            assert (task_folder / "task.md").exists()
            assert (task_folder / "memory").exists()
            assert (task_folder / "memory").is_dir()
    
    def test_create_task_with_special_characters(self):
        """Test task creation with special characters in name"""
        with patch('create_task.get_project_root', return_value=Path(self.temp_dir)):
            result = create_task("Revenue & Profit Analysis (Q4)!", "Q4 revenue analysis")
            
            assert result["success"] is True
            assert result["task_name"] == "revenue___profit_analysis__q4"
            
    def test_create_task_empty_name(self):
        """Test task creation with empty name after sanitization"""
        with patch('create_task.get_project_root', return_value=Path(self.temp_dir)):
            result = create_task("!!!", "Invalid name task")
            
            assert result["success"] is False
            assert "cannot be empty after sanitization" in result["error"]
    
    def test_create_task_duplicate(self):
        """Test creating duplicate task"""
        with patch('create_task.get_project_root', return_value=Path(self.temp_dir)):
            # Create first task
            result1 = create_task("Duplicate Task", "First task")
            assert result1["success"] is True
            
            # Try to create duplicate
            result2 = create_task("Duplicate Task", "Second task")
            assert result2["success"] is False
            assert "already exists" in result2["error"]
    
    def test_task_md_content(self):
        """Test that task.md file has correct content"""
        with patch('create_task.get_project_root', return_value=Path(self.temp_dir)):
            result = create_task("Content Test", "Test task description")
            
            assert result["success"] is True, f"Task creation failed: {result}"
            
            task_folder = Path(self.temp_dir) / "tasks" / result["task_name"]
            task_md = task_folder / "task.md"
            
            assert task_md.exists(), f"task.md file not found at {task_md}"
            
            content = task_md.read_text()
            assert "# Task: Content Test" in content
            assert "Test task description" in content
            assert "Created" in content
            assert "## Status" in content


class TestContextDumper:
    """Test the context dumping functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test task folder
        self.task_folder = Path(self.temp_dir) / "tasks" / "test_task"
        self.task_folder.mkdir(parents=True)
        
    def teardown_method(self):
        """Clean up test environment"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_dump_session_context_success(self):
        """Test successful context dumping"""
        with patch('context_dumper.get_project_root', return_value=Path(self.temp_dir)):
            session_state = {
                "task": "test_task",
                "high_level_plan_status": "planning",
                "data_files": ["customers.csv", "orders.csv"],
                "metadata": {"customers.csv": {"rows": 1000, "columns": 5}}
            }
            
            result = dump_session_context(session_state, "test_task")
            
            assert result["success"] is True
            assert result["variables_saved"] == 4
            assert "current_context.md" in result["context_file"]
            
            # Check file was created
            context_file = self.task_folder / "current_context.md"
            assert context_file.exists()
            
            # Check content
            content = context_file.read_text()
            assert "# Current Session Context" in content
            assert "test_task" in content
            assert "planning" in content
    
    def test_dump_context_nonexistent_task(self):
        """Test dumping context to non-existent task folder"""
        with patch('context_dumper.get_project_root', return_value=Path(self.temp_dir)):
            session_state = {"test": "data"}
            
            result = dump_session_context(session_state, "nonexistent_task")
            
            assert result["success"] is False
            assert "does not exist" in result["error"]
    
    def test_load_session_context_success(self):
        """Test successful context loading"""
        with patch('context_dumper.get_project_root', return_value=Path(self.temp_dir)):
            # First dump some context
            session_state = {"task": "test_task", "status": "active"}
            dump_result = dump_session_context(session_state, "test_task")
            assert dump_result["success"] is True
            
            # Then load it
            load_result = load_session_context("test_task")
            
            assert load_result["success"] is True
            assert "current_context.md" in load_result["context_file"]
            assert "test_task" in load_result["content"]
    
    def test_load_context_no_file(self):
        """Test loading context when file doesn't exist"""
        with patch('context_dumper.get_project_root', return_value=Path(self.temp_dir)):
            result = load_session_context("test_task")
            
            assert result["success"] is False
            assert "Context file not found" in result["error"]


class TestPromptProviders:
    """Test the prompt provider functionality"""
    
    def setup_method(self):
        """Set up test environment with mock prompt files"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock prompt structure
        prompts_dir = Path(self.temp_dir) / "prompts"
        
        # Root agent card
        root_card = prompts_dir / "root_agent_card"
        root_card.mkdir(parents=True)
        
        (root_card / "specs.md").write_text("""# Root Agent Specifications

- **Name**: FPA Assistant
- **Model Provider**: google
- **Model**: gemini-2.5-flash
- **Temperature**: 0
- **Max Tokens**: 4000
""")
        
        (root_card / "description.md").write_text("Test FPA Assistant Description")
        (root_card / "system_prompt.md").write_text("Test system prompt content")
        (root_card / "instruction.md").write_text("Test instruction content")
        
        # High-level planner card
        planner_card = prompts_dir / "high_level_planner_card"
        planner_card.mkdir(parents=True)
        
        (planner_card / "specs.md").write_text("""# Planner Specifications

- **Name**: Strategic Planner
- **Model Provider**: google
- **Model**: gemini-2.5-flash
- **Temperature**: 0
""")
        
        (planner_card / "description.md").write_text("Test Strategic Planner Description")
        (planner_card / "system_prompt.md").write_text("Test planner system prompt")
        (planner_card / "instruction.md").write_text("Test planner instruction")
        
    def teardown_method(self):
        """Clean up test environment"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_load_prompt_file(self):
        """Test loading individual prompt files"""
        with patch('prompt_providers.get_project_root', return_value=Path(self.temp_dir)):
            content = load_prompt_file("root_agent_card", "description.md")
            assert content == "Test FPA Assistant Description"
            
            # Test non-existent file
            content = load_prompt_file("root_agent_card", "nonexistent.md")
            assert content is None
    
    def test_parse_specs_file(self):
        """Test parsing specs file content"""
        specs_content = """# Test Specs

- **Name**: Test Agent
- **Model Provider**: google
- **Temperature**: 0
- **Max Tokens**: 2000
"""
        
        config = parse_specs_file(specs_content)
        
        assert config["name"] == "Test Agent"
        assert config["model_provider"] == "google"
        assert config["temperature"] == 0
        assert config["max_tokens"] == 2000
    
    def test_root_agent_prompt_provider(self):
        """Test root agent configuration loading"""
        with patch('prompt_providers.get_project_root', return_value=Path(self.temp_dir)):
            config = root_agent_prompt_provider()
            
            assert config["name"] == "FPA Assistant"
            assert config["model"]["provider"] == "google"
            assert config["model"]["model"] == "gemini-2.5-flash"
            assert config["model"]["temperature"] == 0
            assert "description" in config["prompts"]
            assert "system_prompt" in config["prompts"]
            assert "create_task" in config["tools"]
    
    def test_high_level_planner_prompt_provider(self):
        """Test planner configuration loading"""
        with patch('prompt_providers.get_project_root', return_value=Path(self.temp_dir)):
            config = high_level_planner_prompt_provider()
            
            assert config["name"] == "Strategic Planner"
            assert config["model"]["provider"] == "google"
            assert "description" in config["prompts"]
            assert "filesystem_operations" in config["tools"]
    
    def test_get_all_agent_configurations(self):
        """Test getting all agent configurations"""
        with patch('prompt_providers.get_project_root', return_value=Path(self.temp_dir)):
            all_configs = get_all_agent_configurations()
            
            assert "root_agent" in all_configs
            assert "high_level_planner" in all_configs
            assert all_configs["root_agent"]["name"] == "FPA Assistant"
            assert all_configs["high_level_planner"]["name"] == "Strategic Planner"
    
    def test_validate_agent_configuration_valid(self):
        """Test validation of valid configuration"""
        valid_config = {
            "name": "Test Agent",
            "model": {"provider": "google"},
            "prompts": {"system_prompt": "test"}
        }
        
        assert validate_agent_configuration(valid_config) is True
    
    def test_validate_agent_configuration_invalid(self):
        """Test validation of invalid configuration"""
        # Missing required fields
        invalid_config = {"name": "Test Agent"}
        assert validate_agent_configuration(invalid_config) is False
        
        # Contains error
        error_config = {"error": "Failed to load", "name": "Test"}
        assert validate_agent_configuration(error_config) is False


def run_all_tests():
    """Run all tests for Phase 1 tools"""
    print("=" * 80)
    print("PHASE 1 TOOLS TEST SUITE")
    print("Testing create_task, context_dumper, and prompt_providers")
    print("=" * 80)
    
    test_classes = [TestCreateTask, TestContextDumper, TestPromptProviders]
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n🧪 Testing {test_class.__name__}...")
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                # Setup
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test
                getattr(test_instance, test_method)()
                print(f"   ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"   ❌ {test_method}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Teardown
                if hasattr(test_instance, 'teardown_method'):
                    try:
                        test_instance.teardown_method()
                    except:
                        pass
    
    print(f"\n" + "=" * 80)
    if passed_tests == total_tests:
        print(f"🎉 ALL TESTS PASSED! ({passed_tests}/{total_tests})")
        print("✅ create_task tool working correctly")
        print("✅ context_dumper tool working correctly") 
        print("✅ prompt_providers working correctly")
        print("✅ Phase 1 tools ready for integration")
    else:
        print(f"❌ {total_tests - passed_tests}/{total_tests} TESTS FAILED")
        print("Phase 1 tools need attention before proceeding")
    
    print("=" * 80)
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)